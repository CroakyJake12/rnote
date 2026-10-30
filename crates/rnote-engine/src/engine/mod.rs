// Modules
pub mod animation;
pub mod config;
pub mod export;
pub mod import;
pub mod rendering;
pub mod snapshot;
pub mod strokecontent;
pub mod visual_debug;

// Re-exports
pub use animation::Animation;
pub use config::EngineConfig;
pub use config::EngineConfigShared;
pub use export::ExportPrefs;
pub use import::ImportPrefs;
pub use snapshot::EngineSnapshot;
pub use strokecontent::StrokeContent;

// Imports
use crate::Image;
use crate::document::Layout;
use crate::pens::PenMode;
use crate::pens::{Pen, PenStyle};
use crate::store::StrokeKey;
use crate::store::render_comp::{self, RenderCompState};
use crate::strokes::content::GeneratedContentImages;
use crate::strokes::textstroke::{TextAttribute, TextStyle};
use crate::{Camera, Document, PenHolder, StrokeStore};
use crate::{SelectionCollision, WidgetFlags};
use futures::StreamExt;
use futures::channel::mpsc::UnboundedReceiver;
use futures::channel::{mpsc, oneshot};
use p2d::bounding_volume::{Aabb, BoundingVolume};
use rnote_compose::eventresult::EventPropagation;
use rnote_compose::ext::AabbExt;
use rnote_compose::penevent::{PenEvent, ShortcutKey};
use rnote_compose::{Color, SplitOrder};
use serde::{Deserialize, Serialize};
use snapshot::Snapshotable;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Instant;
use tracing::error;

/// An immutable view into the engine, excluding the penholder.
#[derive(Debug)]
pub struct EngineView<'a> {
    pub tasks_tx: EngineTaskSender,
    pub config: &'a EngineConfig,
    pub document: &'a Document,
    pub store: &'a StrokeStore,
    pub camera: &'a Camera,
    #[cfg(feature = "ui")]
    pub audioplayer: &'a Option<crate::AudioPlayer>,
    pub animation: &'a Animation,
}

/// Constructs an `EngineView` from an identifier containing an `Engine` instance.
#[macro_export]
macro_rules! engine_view {
    ($engine:ident) => {
        $crate::engine::EngineView {
            tasks_tx: $engine.tasks_tx.clone(),
            config: &$engine.config.read(),
            document: &$engine.document,
            store: &$engine.store,
            camera: &$engine.camera,
            #[cfg(feature = "ui")]
            audioplayer: &$engine.audioplayer,
            animation: &$engine.animation,
        }
    };
}

/// A mutable view into the engine, excluding the penholder.
#[derive(Debug)]
pub struct EngineViewMut<'a> {
    pub tasks_tx: EngineTaskSender,
    pub config: &'a mut EngineConfig,
    pub document: &'a mut Document,
    pub store: &'a mut StrokeStore,
    pub camera: &'a mut Camera,
    #[cfg(feature = "ui")]
    pub audioplayer: &'a mut Option<crate::AudioPlayer>,
    pub animation: &'a mut Animation,
}

/// Constructs an `EngineViewMut` from an identifier containing an `Engine` instance.
#[macro_export]
macro_rules! engine_view_mut {
    ($engine:ident) => {
        $crate::engine::EngineViewMut {
            tasks_tx: $engine.tasks_tx.clone(),
            config: &mut $engine.config.write(),
            document: &mut $engine.document,
            store: &mut $engine.store,
            camera: &mut $engine.camera,
            #[cfg(feature = "ui")]
            audioplayer: &mut $engine.audioplayer,
            animation: &mut $engine.animation,
        }
    };
}

impl EngineViewMut<'_> {
    // Converts itself to the immutable view.
    pub(crate) fn as_im<'m>(&'m self) -> EngineView<'m> {
        EngineView::<'m> {
            tasks_tx: self.tasks_tx.clone(),
            config: self.config,
            document: self.document,
            store: self.store,
            camera: self.camera,
            #[cfg(feature = "ui")]
            audioplayer: self.audioplayer,
            animation: self.animation,
        }
    }
}

#[derive(Debug, Clone)]
/// An engine task, usually coming from a spawned thread and to be processed with [Engine::handle_engine_task].
pub enum EngineTask {
    /// Replace the images for rendering of the given stroke.
    ///
    /// The state of the render component should be set **before** spawning a thread, generating images and sending this task,
    /// to avoid spawning large amounts of already outdated rendering tasks when checking the render component's state on resize/zooming, etc. .
    UpdateStrokeWithImages {
        /// The stroke key.
        key: StrokeKey,
        /// The generated images.
        images: GeneratedContentImages,
        /// The image scale-factor the render task was using while generating the images.
        image_scale: f64,
    },
    /// Appends the images to the rendering of the given stroke.
    ///
    /// The state of the render component should be set **before** spawning a thread, generating images and sending this task,
    /// to avoid spawning large amounts of already outdated rendering tasks when checking the render component's state on resize/zooming, etc. .
    AppendImagesToStroke {
        /// The stroke key
        key: StrokeKey,
        /// The generated images
        images: GeneratedContentImages,
    },
    /// Requests that the typewriter cursor should be blinked/toggled
    BlinkTypewriterCursor,
    /// Change the permanent zoom to the given value
    Zoom(f64),
    /// Indicates that the application is quitting. Sent to quit the handler which receives the tasks.
    Quit,
}

#[derive(Debug, Clone)]
pub struct EngineTaskSender(mpsc::UnboundedSender<EngineTask>);

impl EngineTaskSender {
    pub fn send(&self, task: EngineTask) {
        if let Err(e) = self.0.unbounded_send(task) {
            let err = format!("{e:?}");
            error!(
                "Failed to send engine task {:?}, Err: {err}",
                e.into_inner()
            );
        }
    }
}

#[derive(Debug)]
pub struct EngineTaskReceiver(mpsc::UnboundedReceiver<EngineTask>);

impl EngineTaskReceiver {
    pub fn recv(&mut self) -> futures::stream::Next<'_, UnboundedReceiver<EngineTask>> {
        self.0.next()
    }
}

/// The engine.
#[derive(Debug, Serialize, Deserialize)]
#[serde(default, rename = "engine")]
pub struct Engine {
    #[serde(rename = "config")]
    pub(crate) config: EngineConfigShared,
    #[serde(rename = "document")]
    pub document: Document,
    #[serde(rename = "store")]
    pub store: StrokeStore,
    #[serde(rename = "camera")]
    pub camera: Camera,
    #[serde(rename = "penholder")]
    pub penholder: PenHolder,

    #[cfg(feature = "ui")]
    #[serde(skip)]
    audioplayer: Option<crate::AudioPlayer>,
    #[serde(skip)]
    pub animation: Animation,
    // the task sender. Must not be modified, only cloned.
    #[serde(skip)]
    tasks_tx: EngineTaskSender,
    #[serde(skip)]
    tasks_rx: Option<EngineTaskReceiver>,
    // Background rendering
    #[serde(skip)]
    background_tile_image: Option<Image>,
    #[cfg(feature = "ui")]
    #[serde(skip)]
    background_rendernodes: Vec<gtk4::gsk::RenderNode>,
    // Origin indicator rendering
    #[serde(skip)]
    origin_indicator_image: Option<Image>,
    #[cfg(feature = "ui")]
    #[serde(skip)]
    origin_indicator_rendernode: Option<gtk4::gsk::RenderNode>,
}

impl Default for Engine {
    fn default() -> Self {
        let (tasks_tx, tasks_rx) = futures::channel::mpsc::unbounded::<EngineTask>();

        Self {
            config: EngineConfigShared(Arc::new(RwLock::new(EngineConfig::default()))),
            document: Document::default(),
            store: StrokeStore::default(),
            camera: Camera::default(),
            penholder: PenHolder::default(),

            #[cfg(feature = "ui")]
            audioplayer: None,
            animation: Animation::default(),
            tasks_tx: EngineTaskSender(tasks_tx),
            tasks_rx: Some(EngineTaskReceiver(tasks_rx)),
            background_tile_image: None,
            #[cfg(feature = "ui")]
            background_rendernodes: Vec::default(),
            origin_indicator_image: None,
            #[cfg(feature = "ui")]
            origin_indicator_rendernode: None,
        }
    }
}

impl Engine {
    pub(crate) const STROKE_BOUNDS_INTERSECTION_TOLERANCE: f64 = 1e-3;

    pub fn install_config(
        &mut self,
        config: &EngineConfigShared,
        data_dir: Option<PathBuf>,
    ) -> WidgetFlags {
        let mut widget_flags = WidgetFlags::default();

        let pen_sounds = config.read().pen_sounds;

        self.config = config.clone();
        self.set_pen_sounds(pen_sounds, data_dir);

        widget_flags |= self
            .penholder
            .reinstall_pen_current_style(&mut engine_view_mut!(self));
        widget_flags |= self.doc_resize_to_fit_content();
        widget_flags.redraw = true;
        widget_flags.refresh_ui = true;
        widget_flags
    }

    pub fn engine_tasks_tx(&self) -> EngineTaskSender {
        self.tasks_tx.clone()
    }

    pub fn take_engine_tasks_rx(&mut self) -> Option<EngineTaskReceiver> {
        self.tasks_rx.take()
    }

    /// Whether pen sounds are enabled.
    pub fn pen_sounds(&self) -> bool {
        self.config.read().pen_sounds
    }

    /// Enables/disables the pen sounds.
    ///
    /// If pen sound should be enabled, the pkg data dir must be provided.
    pub fn set_pen_sounds(&mut self, pen_sounds: bool, pkg_data_dir: Option<PathBuf>) {
        self.config.write().pen_sounds = pen_sounds;

        #[cfg(feature = "ui")]
        {
            if pen_sounds {
                if let Some(pkg_data_dir) = pkg_data_dir {
                    // Only create and init a new audioplayer if it does not already exist
                    if self.audioplayer.is_none() {
                        self.audioplayer = match crate::AudioPlayer::new_init(pkg_data_dir) {
                            Ok(audioplayer) => Some(audioplayer),
                            Err(e) => {
                                error!(
                                    "Creating a new audioplayer failed while enabling pen sounds, Err: {e:?}"
                                );
                                None
                            }
                        }
                    }
                }
            } else {
                self.audioplayer.take();
            }
        }
        #[cfg(not(feature = "ui"))]
        {
            let _ = pkg_data_dir;
        }
    }

    pub fn optimize_epd(&self) -> bool {
        self.config.read().optimize_epd
    }

    /// Takes a snapshot of the current state.
    pub fn take_snapshot(&self) -> EngineSnapshot {
        let mut store_history_entry = self.store.create_history_entry();

        // Remove all trashed strokes
        let trashed_keys = store_history_entry
            .trash_components
            .iter()
            .filter_map(|(key, trash_comp)| if trash_comp.trashed { Some(key) } else { None })
            .collect::<Vec<StrokeKey>>();

        for key in trashed_keys {
            Arc::make_mut(&mut store_history_entry.stroke_components).remove(key);
        }

        EngineSnapshot {
            document: self.document.extract_snapshot_data(),
            camera: self.camera.extract_snapshot_data(),
            stroke_components: Arc::clone(&store_history_entry.stroke_components),
            chrono_components: Arc::clone(&store_history_entry.chrono_components),
            chrono_counter: store_history_entry.chrono_counter,
        }
    }

    /// Imports an engine snapshot. A save file should always be loaded with this method.
    pub fn load_snapshot(&mut self, snapshot: EngineSnapshot) -> WidgetFlags {
        self.document = snapshot.document.extract_snapshot_data();
        self.camera = snapshot.camera.extract_snapshot_data();
        let mut widget_flags = self.store.import_from_snapshot(&snapshot)
            | self.doc_resize_autoexpand()
            | self.current_pen_update_state()
            | self.background_rendering_regenerate()
            | self.update_content_rendering_current_viewport();
        widget_flags.refresh_ui = true;
        widget_flags.view_modified = true;
        widget_flags
    }

    /// Records the current store state and saves it as a history entry.
    pub fn record(&mut self, now: Instant) -> WidgetFlags {
        self.store.record(now)
    }

    /// Update the state of the latest history entry with the current document state.
    pub fn update_latest_history_entry(&mut self, now: Instant) -> WidgetFlags {
        self.store.update_latest_history_entry(now)
    }

    /// Undo the latest changes.
    pub fn undo(&mut self, now: Instant) -> WidgetFlags {
        self.store.undo(now)
            | self.doc_resize_autoexpand()
            | self.current_pen_update_state()
            | self.update_rendering_current_viewport()
    }

    /// Redo the latest changes.
    pub fn redo(&mut self, now: Instant) -> WidgetFlags {
        self.store.redo(now)
            | self.doc_resize_autoexpand()
            | self.current_pen_update_state()
            | self.update_rendering_current_viewport()
    }

    pub fn can_undo(&self) -> bool {
        self.store.can_undo()
    }

    pub fn can_redo(&self) -> bool {
        self.store.can_redo()
    }

    // Clears the entire engine.
    pub fn clear(&mut self) -> WidgetFlags {
        self.store.clear() | self.current_pen_update_state() | self.return_to_origin(None)
    }

    /// Handle a received task from tasks_rx.
    /// Returns [WidgetFlags] to indicate what needs to be updated in the UI.
    ///
    /// An example of how to use it:
    /// ```rust, ignore
    ///
    /// glib::spawn_future_local(clone!(@weak canvas, @weak appwindow => async move {
    ///    let mut task_rx = canvas.engine_mut().regenerate_channel();
    ///
    ///    loop {
    ///        if let Some(task) = task_rx.next().await {
    ///            let (widget_flags, quit) = canvas.engine_mut().handle_engine_task(task);
    ///            canvas.emit_handle_widget_flags(widget_flags);
    ///
    ///            if quit {
    ///                break;
    ///            }
    ///        }
    ///    }
    /// }));
    /// ```
    pub fn handle_engine_task(&mut self, task: EngineTask) -> (WidgetFlags, bool) {
        let mut widget_flags = WidgetFlags::default();
        let mut quit = false;

        match task {
            EngineTask::UpdateStrokeWithImages {
                key,
                images,
                image_scale,
            } => {
                if let Some(state) = self.store.render_comp_state(key) {
                    match state {
                        RenderCompState::Complete | RenderCompState::ForViewport(_) => {
                            // The rendering was already regenerated in the meantime,
                            // so we just discard the render task result
                        }
                        RenderCompState::BusyRenderingInTask => {
                            if (self.camera.image_scale()
                                - render_comp::RENDER_IMAGE_SCALE_TOLERANCE
                                ..self.camera.image_scale()
                                    + render_comp::RENDER_IMAGE_SCALE_TOLERANCE)
                                .contains(&image_scale)
                            {
                                // Only when the image scale is roughly the same as when the render task was started,
                                // the new images are considered valid and can replace the old.
                                self.store.replace_rendering_with_images(key, images);
                            }
                            widget_flags.redraw = true;
                        }
                        RenderCompState::Dirty => {
                            // If the state was flagged dirty in the meantime,
                            // it is expected that retriggering rendering will be handled elsewhere
                        }
                    }
                }
            }
            EngineTask::AppendImagesToStroke { key, images } => {
                self.store.append_rendering_images(key, images);
                widget_flags.redraw = true;
            }
            EngineTask::BlinkTypewriterCursor => {
                if let Pen::Typewriter(typewriter) = self.penholder.current_pen_mut() {
                    typewriter.toggle_cursor_visibility();
                    widget_flags.redraw = true;
                }
            }
            EngineTask::Zoom(zoom) => {
                widget_flags |= self.camera.zoom_temporarily_to(1.0) | self.camera.zoom_to(zoom);

                let all_strokes = self.store.stroke_keys_unordered();
                self.store.set_rendering_dirty_for_strokes(&all_strokes);
                widget_flags |= self.doc_resize_autoexpand()
                    | self.background_rendering_regenerate()
                    | self.update_rendering_current_viewport();
            }
            EngineTask::Quit => {
                widget_flags |= self.set_active(false);
                quit = true;
            }
        }

        (widget_flags, quit)
    }

    /// Handle a pen event.
    pub fn handle_pen_event(
        &mut self,
        event: PenEvent,
        pen_mode: Option<PenMode>,
        now: Instant,
    ) -> (EventPropagation, WidgetFlags) {
        self.penholder
            .handle_pen_event(event, pen_mode, now, &mut engine_view_mut!(self))
    }

    /// Handle a pressed shortcut key.
    pub fn handle_pressed_shortcut_key(
        &mut self,
        shortcut_key: ShortcutKey,
        now: Instant,
    ) -> (EventPropagation, WidgetFlags) {
        self.penholder
            .handle_pressed_shortcut_key(shortcut_key, now, &mut engine_view_mut!(self))
    }

    /// Change the pen style.
    pub fn change_pen_style(&mut self, new_style: PenStyle) -> WidgetFlags {
        self.penholder
            .change_style(new_style, &mut engine_view_mut!(self))
    }

    /// Change the pen style (temporary) override.
    pub fn change_pen_style_override(
        &mut self,
        new_style_override: Option<PenStyle>,
    ) -> WidgetFlags {
        self.penholder
            .change_style_override(new_style_override, &mut engine_view_mut!(self))
    }

    /// Change the pen mode. Relevant for stylus input.
    pub fn change_pen_mode(&mut self, pen_mode: PenMode) -> WidgetFlags {
        self.penholder
            .change_pen_mode(pen_mode, &mut engine_view_mut!(self))
    }

    /// Reinstall the pen in the current style.
    pub fn reinstall_pen_current_style(&mut self) -> WidgetFlags {
        self.penholder
            .reinstall_pen_current_style(&mut engine_view_mut!(self))
    }

    /// Set the engine active or inactive.
    pub fn set_active(&mut self, active: bool) -> WidgetFlags {
        let mut widget_flags = WidgetFlags::default();
        if active {
            widget_flags |= self.reinstall_pen_current_style()
                | self.background_rendering_regenerate()
                | self.update_content_rendering_current_viewport();
            widget_flags.view_modified = true;
        } else {
            widget_flags |= self.clear_rendering() | self.penholder.deinit_current_pen();
        }
        widget_flags
    }

    /// Generate bounds for each page on the document which contains content.
    pub fn pages_bounds_w_content(&self, split_order: SplitOrder) -> Vec<Aabb> {
        let doc_bounds = self.document.bounds();
        let keys = self.store.stroke_keys_as_rendered();

        let strokes_bounds = self.store.strokes_bounds(&keys);

        let pages_bounds = doc_bounds
            .split_extended_origin_aligned(self.document.config.format.size(), split_order)
            .into_iter()
            .filter(|page_bounds| {
                // Filter the pages out that don't intersect with any stroke
                strokes_bounds.iter().any(|stroke_bounds| {
                    stroke_bounds.intersects_w_tolerance(
                        page_bounds,
                        Self::STROKE_BOUNDS_INTERSECTION_TOLERANCE,
                    )
                })
            })
            .collect::<Vec<Aabb>>();

        if pages_bounds.is_empty() {
            // If no page has content, return the origin page
            vec![Aabb::new(
                na::point![0.0, 0.0],
                self.document.config.format.size().into(),
            )]
        } else {
            pages_bounds
        }
    }

    /// Generates bounds which contain all pages on the doc with content, extended to fit the current format.
    pub fn bounds_w_content_extended(&self) -> Option<Aabb> {
        let pages_bounds = self.pages_bounds_w_content(SplitOrder::default());

        if pages_bounds.is_empty() {
            return None;
        }

        Some(
            pages_bounds
                .into_iter()
                .fold(Aabb::new_invalid(), |prev, next| prev.merged(&next)),
        )
    }

    /// First zoom temporarily and then permanently after a timeout.
    ///
    /// Repeated calls to this function reset the timeout.
    pub fn zoom_w_timeout(&mut self, zoom: f64) -> WidgetFlags {
        self.camera.zoom_w_timeout(zoom, self.tasks_tx.clone())
    }

    pub fn set_scale_factor(&mut self, scale_factor: f64) -> WidgetFlags {
        self.store
            .set_rendering_dirty_for_strokes(&self.store.stroke_keys_as_rendered());
        self.camera.set_scale_factor(scale_factor)
            | self.background_rendering_regenerate()
            | self.update_content_rendering_current_viewport()
    }

    /// Resizes the doc to the format and to fit all strokes.
    ///
    /// Background rendering then needs to be updated.
    pub fn doc_resize_to_fit_content(&mut self) -> WidgetFlags {
        let widget_flags = self
            .document
            .resize_to_fit_content(&self.store, &self.camera);
        widget_flags | self.update_rendering_current_viewport()
    }

    pub fn return_to_origin(&mut self, parent_width: Option<f64>) -> WidgetFlags {
        let zoom = self.camera.zoom();
        let new_offset = if let Some(parent_width) = parent_width {
            if self.document.config.format.width() * zoom <= parent_width {
                na::vector![
                    (self.document.config.format.width() * 0.5 * zoom) - parent_width * 0.5,
                    -Document::SHADOW_WIDTH * zoom
                ]
            } else {
                // If the zoomed format width is larger than the displayed surface, we zoom to a fixed origin
                na::vector![
                    -Document::SHADOW_WIDTH * zoom,
                    -Document::SHADOW_WIDTH * zoom
                ]
            }
        } else {
            na::vector![
                -Document::SHADOW_WIDTH * zoom,
                -Document::SHADOW_WIDTH * zoom
            ]
        };
        self.camera_set_offset_expand(new_offset)
    }

    /// Resize the doc when in autoexpanding layouts. called e.g. when finishing a new stroke.
    ///
    /// Background rendering then needs to be updated.
    pub fn doc_resize_autoexpand(&mut self) -> WidgetFlags {
        let widget_flags = self.document.resize_autoexpand(&self.store, &self.camera);
        widget_flags | self.update_rendering_current_viewport()
    }

    /// Expand the doc to the camera when in autoexpanding layouts. called e.g. when dragging with touch.
    ///
    /// Background and content rendering then need to be updated.
    pub fn doc_expand_autoexpand(&mut self) -> WidgetFlags {
        self.document.expand_autoexpand(&self.camera, &self.store)
    }

    /// Add a page to the document when in fixed size layout.
    ///
    /// Document layout must be set to fixed-size.
    pub fn doc_add_page_fixed_size(&mut self) -> WidgetFlags {
        let mut widget_flags = WidgetFlags::default();
        if self.document.add_page_fixed_size() {
            widget_flags |= self.update_rendering_current_viewport();
            widget_flags.resize = true;
        }
        widget_flags
    }

    /// Remove a page from the document when in fixed size layout.
    ///
    /// Document layout must be set to fixed-size.
    pub fn doc_remove_page_fixed_size(&mut self) -> WidgetFlags {
        let mut widget_flags = WidgetFlags::default();
        if self.document.remove_page_fixed_size() {
            self.store.set_trashed_keys(
                &self
                    .store
                    .keys_below_y(self.document.y + self.document.height),
                true,
            );
            widget_flags |= self.record(Instant::now()) | self.update_rendering_current_viewport();
            widget_flags.resize = true;
        }
        widget_flags
    }

    /// Update the viewport offset of the camera, clamped to mins and maxs values depending on the document layout.
    ///
    /// Background and content rendering then need to be updated.
    pub fn camera_set_offset(&mut self, offset: na::Vector2<f64>) -> WidgetFlags {
        self.camera.set_offset(offset, &self.document)
    }

    /// Update the viewport offset of the camera, clamped to mins and maxs values depending on the document layout.
    ///
    /// Expands the document when in autoexpanding layouts.
    ///
    /// Background and content rendering then need to be updated.
    pub fn camera_set_offset_expand(&mut self, offset: na::Vector2<f64>) -> WidgetFlags {
        let widget_flags = self.camera.set_offset(offset, &self.document);
        widget_flags | self.doc_expand_autoexpand()
    }

    /// Update the viewport size of the camera.
    ///
    /// Background and content rendering then need to be updated.
    pub fn camera_set_size(&mut self, size: na::Vector2<f64>) -> WidgetFlags {
        self.camera.set_size(size)
    }

    /// The minimum and maximum surface bounds (document including overshoot) in surface coordinate space.
    pub fn camera_surface_mins_maxs(&self) -> (na::Vector2<f64>, na::Vector2<f64>) {
        self.camera.surface_mins_maxs(&self.document)
    }

    /// Update the current pen with the current engine state.
    ///
    /// Needs to be called when the engine state was changed outside of pen events.
    /// ( e.g. trash all strokes, set strokes selected, etc. )
    pub fn current_pen_update_state(&mut self) -> WidgetFlags {
        self.penholder
            .current_pen_update_state(&mut engine_view_mut!(self))
    }

    /// Fetch clipboard content from the current pen.
    #[allow(clippy::type_complexity)]
    pub fn fetch_clipboard_content(
        &self,
    ) -> oneshot::Receiver<anyhow::Result<(Vec<(Vec<u8>, String)>, WidgetFlags)>> {
        self.penholder.fetch_clipboard_content(&engine_view!(self))
    }

    /// Cut clipboard content from the current pen.
    #[allow(clippy::type_complexity)]
    pub fn cut_clipboard_content(
        &mut self,
    ) -> oneshot::Receiver<anyhow::Result<(Vec<(Vec<u8>, String)>, WidgetFlags)>> {
        self.penholder
            .cut_clipboard_content(&mut engine_view_mut!(self))
    }

    pub fn set_doc_layout(&mut self, layout: Layout) -> WidgetFlags {
        if self.document.config.layout != layout {
            self.document.config.layout = layout;
            self.doc_resize_to_fit_content()
        } else {
            self.doc_resize_autoexpand()
        }
    }

    pub fn select_all_strokes(&mut self) -> WidgetFlags {
        let widget_flags = self.change_pen_style(PenStyle::Selector);
        self.store
            .set_selected_keys(&self.store.stroke_keys_as_rendered(), true);
        widget_flags
            | self.current_pen_update_state()
            | self.doc_resize_autoexpand()
            | self.record(Instant::now())
            | self.update_rendering_current_viewport()
    }

    pub fn deselect_all_strokes(&mut self) -> WidgetFlags {
        let widget_flags = self.change_pen_style(PenStyle::Selector);
        self.store
            .set_selected_keys(&self.store.selection_keys_as_rendered(), false);
        widget_flags
            | self.current_pen_update_state()
            | self.doc_resize_autoexpand()
            | self.record(Instant::now())
            | self.update_rendering_current_viewport()
    }

    pub fn select_with_bounds(
        &mut self,
        bounds: Aabb,
        collision: SelectionCollision,
    ) -> WidgetFlags {
        let select = match collision {
            SelectionCollision::Contains => self.store.stroke_keys_as_rendered_in_bounds(bounds),
            SelectionCollision::Intersects => self
                .store
                .stroke_keys_as_rendered_intersecting_bounds(bounds),
        };
        self.store.set_selected_keys(&select, true);
        self.doc_resize_autoexpand()
            | self.record(Instant::now())
            | self.update_rendering_current_viewport()
    }

    pub fn duplicate_selection(&mut self) -> WidgetFlags {
        let new_selected = self.store.duplicate_selection();
        self.store.update_geometry_for_strokes(&new_selected);
        self.current_pen_update_state()
            | self.doc_resize_autoexpand()
            | self.record(Instant::now())
            | self.update_rendering_current_viewport()
    }


    /// Drain real queued headless render results before a keyed mutation.
    /// A deadline/budget failure refuses the next edit; no fake render completion.
    /// The receiver remains owned by this Engine on every return path.
    pub fn settle_headless_render_tasks(&mut self) -> anyhow::Result<()> {
        let mut receiver = self.tasks_rx.take()
            .ok_or_else(|| anyhow::anyhow!("The original engine task receiver is not owned here"))?;
        let result = (|| {
            let deadline = Instant::now() + std::time::Duration::from_secs(5);
            let mut consumed = 0usize;
            loop {
                loop {
                    #[allow(deprecated)]
                    match receiver.0.try_next() {
                        Ok(Some(task)) => {
                            consumed += 1;
                            if consumed > 65536 {
                                anyhow::bail!("Headless render task budget exceeded");
                            }
                            if self.handle_engine_task(task).1 {
                                anyhow::bail!("Original engine requested shutdown");
                            }
                        }
                        Ok(None) => anyhow::bail!("Original engine task channel closed"),
                        Err(_) => break,
                    }
                }
                let busy = self.store.stroke_keys_unordered().into_iter().any(|key|
                    matches!(self.store.render_comp_state(key), Some(RenderCompState::BusyRenderingInTask)));
                if !busy { return Ok(()); }
                if Instant::now() >= deadline {
                    anyhow::bail!("Original native render tasks did not settle before the deadline");
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        })();
        self.tasks_rx = Some(receiver);
        result
    }

    /// A bounded headless invocation of the genuine donor SplitColliding path.
    /// Caller uses a detached Engine; the original document/history never change.
    pub fn split_strokes_traced_at(
        &mut self,
        element: rnote_compose::penpath::Element,
        width: f64,
        now: Instant,
    ) -> anyhow::Result<Vec<crate::store::trash_comp::SplitStrokeLineage>> {
        if !width.is_finite() || !(1.0..=500.0).contains(&width)
            || !element.pos.iter().all(|value| value.is_finite())
            || !element.pressure.is_finite() || !(0.0..=1.0).contains(&element.pressure)
            || matches!(self.penholder.current_pen_progress(), rnote_compose::penevent::PenProgress::InProgress) {
            anyhow::bail!("Invalid or unfinished native split gesture");
        }
        self.settle_headless_render_tasks()?;
        let config = crate::pens::pensconfig::eraserconfig::EraserConfig {
            width,
            style: crate::pens::pensconfig::eraserconfig::EraserStyle::SplitCollidingStrokes,
        };
        let bounds = config.eraser_bounds(element);
        if !bounds.mins.iter().chain(bounds.maxs.iter()).all(|value| value.is_finite()) {
            anyhow::bail!("Split eraser bounds overflow");
        }
        // The immutable proposal covers all document ink under the exact eraser
        // bounds, independently of an offscreen/default headless camera viewport.
        let (modified, flags, lineage) = self.store.split_colliding_strokes_traced(bounds, bounds);
        if flags.store_modified {
            self.store.update_geometry_for_strokes(&modified);
            self.store.regenerate_rendering_for_strokes(&modified, bounds, self.camera.image_scale());
            let _ = self.current_pen_update_state() | self.doc_resize_autoexpand() | self.record(now);
        }
        Ok(lineage)
    }

    /// Trash exactly one live native key using the normal donor history.
    /// The caller owns exclusive engine/task access; this is not a Home grant.
    /// Busy rendering and active pen gestures refuse before any store mutation.
    /// Returned flags request rendering; they are not a rendered-frame receipt.
    pub fn trash_stroke_key(
        &mut self,
        key: StrokeKey,
        now: Instant,
    ) -> anyhow::Result<WidgetFlags> {
        self.require_editable_stroke_key(key)?;
        self.store.set_trashed_keys(&[key], true);
        let mut flags = self.current_pen_update_state()
            | self.doc_resize_autoexpand()
            | self.record(now)
            | self.update_rendering_current_viewport();
        flags.store_modified = true;
        flags.redraw = true;
        Ok(flags)
    }

    /// Translate one live key without changing its identity, neighbours or layer.
    /// None denotes an exact zero delta; it does not record or truncate redo.
    /// All numerical/admission checks finish on a detached stroke clone first.
    pub fn translate_stroke_key(
        &mut self,
        key: StrokeKey,
        offset: na::Vector2<f64>,
        now: Instant,
    ) -> anyhow::Result<Option<WidgetFlags>> {
        use crate::strokes::Content;
        use rnote_compose::shapes::Shapeable;
        use rnote_compose::transform::Transformable;

        self.require_editable_stroke_key(key)?;
        if !offset.iter().all(|value| value.is_finite()) {
            anyhow::bail!("Native stroke translation must be finite");
        }
        if offset == na::Vector2::zeros() {
            return Ok(None);
        }
        let mut candidate = self.store.get_stroke_ref(key)
            .ok_or_else(|| anyhow::anyhow!("Native stroke key is unavailable"))?.clone();
        Self::check_keyed_translation_bounds(candidate.bounds(), offset)?;
        candidate.translate(offset);
        candidate.update_geometry();
        for bounds in std::iter::once(candidate.bounds()).chain(candidate.hitboxes()) {
            for axis in 0..2 {
                if !bounds.mins[axis].is_finite() || !bounds.maxs[axis].is_finite()
                    || bounds.mins[axis] > bounds.maxs[axis] {
                    anyhow::bail!("Native stroke translation has invalid donor geometry");
                }
            }
        }

        // Reuse the donor's live COW mutation, spatial index and geometry cache.
        // Never load an EngineSnapshot to simulate a live edit: that resets undo.
        self.store.translate_strokes(&[key], offset);
        self.store.update_geometry_for_strokes(&[key]);
        let mut flags = self.current_pen_update_state()
            | self.doc_resize_autoexpand()
            | self.record(now)
            | self.update_rendering_current_viewport();
        flags.store_modified = true;
        flags.redraw = true;
        Ok(Some(flags))
    }

    fn require_editable_stroke_key(&self, key: StrokeKey) -> anyhow::Result<()> {
        if matches!(self.penholder.current_pen_progress(), rnote_compose::penevent::PenProgress::InProgress) {
            anyhow::bail!("Finish the original pen gesture before a keyed edit");
        }
        if self.store.get_stroke_ref(key).is_none() || self.store.trashed(key) != Some(false) {
            anyhow::bail!("Native stroke key is unknown, stale or trashed");
        }
        match self.store.render_comp_state(key) {
            Some(RenderCompState::BusyRenderingInTask) | None =>
                anyhow::bail!("Settle the original render task before a keyed edit"),
            _ => Ok(()),
        }
    }

    fn check_keyed_translation_bounds(bounds: Aabb, offset: na::Vector2<f64>) -> anyhow::Result<()> {
        for axis in 0..2 {
            let old_min = bounds.mins[axis];
            let old_max = bounds.maxs[axis];
            let new_min = old_min + offset[axis];
            let new_max = old_max + offset[axis];
            if !old_min.is_finite() || !old_max.is_finite() || old_min > old_max
                || !new_min.is_finite() || !new_max.is_finite()
                || (old_max > old_min && new_max <= new_min) {
                anyhow::bail!("Native stroke translation overflows or loses representable geometry");
            }
        }
        Ok(())
    }

    pub fn trash_selection(&mut self) -> WidgetFlags {
        let selection_keys = self.store.selection_keys_as_rendered();
        self.store.set_trashed_keys(&selection_keys, true);
        self.current_pen_update_state()
            | self.doc_resize_autoexpand()
            | self.record(Instant::now())
            | self.update_rendering_current_viewport()
    }

    pub fn nothing_selected(&self) -> bool {
        self.store.selection_keys_unordered().is_empty()
    }

    pub fn change_selection_stroke_colors(&mut self, stroke_color: Color) -> WidgetFlags {
        self.store
            .change_stroke_colors(&self.store.selection_keys_as_rendered(), stroke_color)
            | self.record(Instant::now())
            | self.update_content_rendering_current_viewport()
    }

    pub fn change_selection_fill_colors(&mut self, fill_color: Color) -> WidgetFlags {
        self.store
            .change_fill_colors(&self.store.selection_keys_as_rendered(), fill_color)
            | self.record(Instant::now())
            | self.update_content_rendering_current_viewport()
    }

    pub fn invert_selection_colors(&mut self) -> WidgetFlags {
        self.store
            .invert_color_brightness(&self.store.selection_keys_as_rendered())
            | self.record(Instant::now())
            | self.update_content_rendering_current_viewport()
    }

    pub fn text_selection_change_style<F>(&mut self, modify_func: F) -> WidgetFlags
    where
        F: FnOnce(&mut TextStyle),
    {
        let mut widget_flags = WidgetFlags::default();
        if let Pen::Typewriter(typewriter) = self.penholder.current_pen_mut() {
            widget_flags |= typewriter
                .change_text_style_in_modifying_stroke(modify_func, &mut engine_view_mut!(self))
        }
        widget_flags
    }

    pub fn text_selection_remove_attributes(&mut self) -> WidgetFlags {
        let mut widget_flags = WidgetFlags::default();
        if let Pen::Typewriter(typewriter) = self.penholder.current_pen_mut() {
            widget_flags |=
                typewriter.remove_text_attributes_current_selection(&mut engine_view_mut!(self))
        }
        widget_flags
    }

    pub fn text_select_closest_word(&mut self) {
        if let Pen::Typewriter(typewriter) = self.penholder.current_pen_mut() {
            typewriter.select_closest_word(&mut engine_view_mut!(self))
        }
    }

    pub fn text_select_closest_line(&mut self) {
        if let Pen::Typewriter(typewriter) = self.penholder.current_pen_mut() {
            typewriter.select_closest_line(&mut engine_view_mut!(self))
        }
    }

    pub fn text_selection_toggle_attribute(
        &mut self,
        text_attribute: TextAttribute,
    ) -> WidgetFlags {
        let mut widget_flags = WidgetFlags::default();
        if let Pen::Typewriter(typewriter) = self.penholder.current_pen_mut() {
            widget_flags |= typewriter.toggle_text_attribute_current_selection(
                text_attribute,
                &mut engine_view_mut!(self),
            )
        }
        widget_flags
    }

    pub fn text_selection_add_attribute(&mut self, text_attribute: TextAttribute) -> WidgetFlags {
        let mut widget_flags = WidgetFlags::default();
        if let Pen::Typewriter(typewriter) = self.penholder.current_pen_mut() {
            widget_flags |= typewriter
                .add_text_attribute_current_selection(text_attribute, &mut engine_view_mut!(self))
        }
        widget_flags
    }

    pub fn text_change_color(&mut self, color: Color) -> WidgetFlags {
        let mut widget_flags = WidgetFlags::default();
        if let Pen::Typewriter(typewriter) = self.penholder.current_pen_mut() {
            if typewriter.selection_range().is_some() {
                widget_flags |= typewriter.replace_text_attribute_current_selection(
                    TextAttribute::TextColor(color),
                    &mut engine_view_mut!(self),
                )
            } else {
                widget_flags |= typewriter.change_text_style_in_modifying_stroke(
                    |style| style.color = color,
                    &mut engine_view_mut!(self),
                )
            }
        }
        widget_flags
    }

    /// Handle a requested animation frame.
    ///
    /// Can request another frame using `EngineViewMut#animation.claim_frame()`.
    pub fn handle_animation_frame(&mut self) {
        self.penholder
            .handle_animation_frame(&mut engine_view_mut!(self));
    }

    pub fn current_pen_style_w_override(&self) -> PenStyle {
        self.penholder
            .current_pen_style_w_override(&engine_view!(self))
    }
}


#[cfg(test)]
mod keyed_stroke_edit_tests {
    use super::*;
    use crate::strokes::{BrushStroke, Stroke};
    use rnote_compose::{PenPath, Style};
    use rnote_compose::penpath::Element;
    use rnote_compose::shapes::Shapeable;
    use slotmap::Key;

    fn brush() -> Stroke {
        let path = PenPath::try_from_elements([
            Element::new(na::vector![10_000.0, 10_000.0], 0.2),
            Element::new(na::vector![10_020.0, 10_010.0], 0.7),
            Element::new(na::vector![10_040.0, 10_005.0], 1.0),
        ]).unwrap();
        Stroke::BrushStroke(BrushStroke::from_penpath(path, Style::default()))
    }

    fn fixture() -> (Engine, StrokeKey, StrokeKey, StrokeKey) {
        let mut engine = Engine::default();
        // Keep cache task production out of these geometry/history controls.
        // Actual viewport rendering/async ownership needs its own native checks.
        engine.camera.set_size(na::vector![1.0, 1.0]);
        engine.camera.set_viewport_center(na::vector![-1_000_000.0, -1_000_000.0]);
        let first = engine.store.insert_stroke(brush(), None);
        let neighbour = engine.store.insert_stroke(brush(), None);
        let stale = engine.store.insert_stroke(brush(), None);
        assert!(engine.store.remove_stroke(stale).is_some());
        engine.record(Instant::now());
        (engine, first, neighbour, stale)
    }

    fn entry(engine: &Engine) -> serde_json::Value {
        serde_json::to_value(engine.store.create_history_entry()).unwrap()
    }

    fn stroke(engine: &Engine, key: StrokeKey) -> serde_json::Value {
        serde_json::to_value(engine.store.get_stroke_ref(key).unwrap()).unwrap()
    }

    #[test]
    fn keyed_trash_preserves_overlapping_neighbour_and_restores_same_key_on_undo_redo() {
        let (mut engine, target, neighbour, stale) = fixture();
        let original = stroke(&engine, target);
        let neighbour_before = stroke(&engine, neighbour);
        assert!(engine.trash_stroke_key(target, Instant::now()).unwrap().store_modified);
        assert!(!engine.take_snapshot().stroke_components.contains_key(target));
        assert!(engine.take_snapshot().stroke_components.contains_key(neighbour));
        assert_eq!(stroke(&engine, neighbour), neighbour_before);
        let deleted = entry(&engine);
        assert!(engine.trash_stroke_key(target, Instant::now()).is_err());
        assert!(engine.trash_stroke_key(stale, Instant::now()).is_err());
        assert!(engine.trash_stroke_key(StrokeKey::null(), Instant::now()).is_err());
        assert_eq!(entry(&engine), deleted);
        assert!(engine.can_undo());
        engine.undo(Instant::now());
        assert!(engine.take_snapshot().stroke_components.contains_key(target));
        assert_eq!(stroke(&engine, target), original);
        assert_eq!(stroke(&engine, neighbour), neighbour_before);
        assert!(engine.can_redo());
        engine.redo(Instant::now());
        assert!(!engine.take_snapshot().stroke_components.contains_key(target));
        assert!(engine.take_snapshot().stroke_components.contains_key(neighbour));
    }

    #[test]
    fn keyed_translate_updates_real_spatial_queries_and_one_history_step() {
        let (mut engine, target, neighbour, _) = fixture();
        let before = stroke(&engine, target);
        let other = stroke(&engine, neighbour);
        let old_bounds = engine.store.get_stroke_ref(target).unwrap().bounds();
        let original_order = engine.store.stroke_keys_as_rendered();
        let delta = na::vector![1_000.0, 700.0];
        assert!(engine.translate_stroke_key(target, delta, Instant::now()).unwrap().unwrap().store_modified);
        let moved_bounds = engine.store.get_stroke_ref(target).unwrap().bounds();
        for axis in 0..2 {
            assert!((moved_bounds.mins[axis] - old_bounds.mins[axis] - delta[axis]).abs() < 1e-8);
            assert!((moved_bounds.maxs[axis] - old_bounds.maxs[axis] - delta[axis]).abs() < 1e-8);
        }
        assert_eq!(engine.store.stroke_keys_as_rendered(), original_order);
        assert_eq!(stroke(&engine, neighbour), other);
        let old_hits = engine.store.stroke_keys_as_rendered_intersecting_bounds(old_bounds);
        assert!(!old_hits.contains(&target));
        assert!(old_hits.contains(&neighbour));
        let new_hits = engine.store.stroke_keys_as_rendered_intersecting_bounds(moved_bounds);
        assert!(new_hits.contains(&target));
        assert!(!new_hits.contains(&neighbour));
        let moved = stroke(&engine, target);
        engine.undo(Instant::now());
        assert_eq!(stroke(&engine, target), before);
        assert_eq!(stroke(&engine, neighbour), other);
        assert!(engine.store.stroke_keys_as_rendered_intersecting_bounds(old_bounds).contains(&target));
        engine.redo(Instant::now());
        assert_eq!(stroke(&engine, target), moved);
        assert_eq!(stroke(&engine, neighbour), other);
    }

    #[test]
    fn rejected_and_zero_edits_preserve_real_state_and_available_redo() {
        let (mut engine, target, _, stale) = fixture();
        engine.translate_stroke_key(target, na::vector![30.0, 40.0], Instant::now()).unwrap().unwrap();
        engine.undo(Instant::now());
        assert!(engine.can_redo());
        let before = entry(&engine);
        assert!(engine.translate_stroke_key(target, na::Vector2::zeros(), Instant::now()).unwrap().is_none());
        for delta in [na::vector![f64::NAN, 0.0], na::vector![0.0, f64::INFINITY], na::vector![f64::MAX, 0.0]] {
            assert!(engine.translate_stroke_key(target, delta, Instant::now()).is_err());
            assert_eq!(entry(&engine), before);
            assert!(engine.can_redo());
        }
        assert!(engine.translate_stroke_key(stale, na::vector![1.0, 1.0], Instant::now()).is_err());
        assert!(engine.trash_stroke_key(stale, Instant::now()).is_err());
        assert_eq!(entry(&engine), before);
        assert!(engine.can_redo());
        engine.redo(Instant::now());
        assert_ne!(entry(&engine), before);
    }

    #[test]
    fn finite_delta_overflow_is_rejected_before_donor_mutation() {
        let bounds = Aabb::new(na::point![f64::MAX * 0.6, 0.0], na::point![f64::MAX * 0.8, 1.0]);
        assert!(Engine::check_keyed_translation_bounds(bounds, na::vector![f64::MAX * 0.6, 0.0]).is_err());
        assert!(Engine::check_keyed_translation_bounds(bounds, na::vector![0.0, 1.0]).is_ok());
    }
}
