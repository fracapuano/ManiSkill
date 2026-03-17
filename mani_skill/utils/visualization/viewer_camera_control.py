from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import sapien
from sapien import internal_renderer as R
from sapien.utils.viewer.plugin import Plugin
from transforms3d.quaternions import mat2quat

from mani_skill.sensors.camera import Camera
from mani_skill.utils.structs.pose import Pose

if TYPE_CHECKING:
    from mani_skill.envs.sapien_env import BaseEnv


_VIEWER_GL_POSE = sapien.Pose([0, 0, 0], [-0.5, -0.5, 0.5, 0.5])
_CAMERA_LINESET_VERTICES = [
    0,
    0,
    0,
    1,
    1,
    -1,
    0,
    0,
    0,
    -1,
    1,
    -1,
    0,
    0,
    0,
    1,
    -1,
    -1,
    0,
    0,
    0,
    -1,
    -1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    1,
    -1,
    1,
    1.2,
    -1,
    0,
    2,
    -1,
    0,
    2,
    -1,
    -1,
    1.2,
    -1,
    -1,
    1.2,
    -1,
    1,
    1.2,
    -1,
]
_INACTIVE_CAMERA_LINESET_COLORS = [0.9254901961, 0.5764705882, 0.1882352941, 1.0] * 22
_ACTIVE_CAMERA_LINESET_COLORS = [0.1882352941, 0.7921568627, 0.3882352941, 1.0] * 22


class ViewerCameraControlPlugin(Plugin):
    """Edit ManiSkill cameras directly from the SAPIEN viewer."""

    def __init__(
        self,
        env: "BaseEnv",
        enabled: bool = True,
        selection_radius_px: float = 18.0,
    ):
        self.env = env
        self.enabled = enabled
        self.selection_radius_px = selection_radius_px
        self._active_camera_name: Optional[str] = None
        self._line_sets = []
        self._line_set_names: list[str] = []
        self._line_set_active_name: Optional[str] = None
        self._line_set_model = None
        self._active_line_set_model = None
        self._ui_window = None
        self._gizmo = None
        self._camera_selector = None

    def init(self, viewer):
        super().init(viewer)
        if hasattr(self.viewer, "register_click_handler"):
            self.viewer.register_click_handler(self._handle_click)
        self._create_visual_models()

    def notify_scene_change(self):
        self._clear_camera_linesets()
        self._ui_window = None
        if self._active_camera_name not in self.camera_names:
            self._active_camera_name = (
                self.camera_names[0] if self.camera_names else None
            )

    def clear_scene(self):
        self._clear_camera_linesets()

    def close(self):
        self._clear_camera_linesets()
        self._line_set_model = None
        self._active_line_set_model = None
        self._ui_window = None
        self._gizmo = None
        self._camera_selector = None

    @property
    def editable_cameras(self) -> dict[str, Camera]:
        cameras = {}
        for uid, camera in self.env._human_render_cameras.items():
            cameras[f"render:{uid}"] = camera
        for uid, sensor in self.env._sensors.items():
            if isinstance(sensor, Camera):
                cameras[f"sensor:{uid}"] = sensor
        return cameras

    @property
    def camera_names(self) -> list[str]:
        return list(self.editable_cameras.keys())

    @property
    def camera_items(self) -> list[str]:
        return ["None"] + self.camera_names

    @property
    def active_camera_name(self) -> Optional[str]:
        if self._active_camera_name not in self.editable_cameras:
            self._active_camera_name = (
                self.camera_names[0] if self.camera_names else None
            )
        return self._active_camera_name

    @active_camera_name.setter
    def active_camera_name(self, value: Optional[str]):
        self._active_camera_name = value if value in self.editable_cameras else None
        self.viewer.notify_render_update()

    @property
    def active_camera_index(self) -> int:
        active_camera_name = self.active_camera_name
        if active_camera_name is None:
            return 0
        return self.camera_names.index(active_camera_name) + 1

    @active_camera_index.setter
    def active_camera_index(self, index: int):
        if index <= 0 or index > len(self.camera_names):
            self.active_camera_name = None
        else:
            self.active_camera_name = self.camera_names[index - 1]

    @property
    def active_camera(self) -> Optional[Camera]:
        active_camera_name = self.active_camera_name
        if active_camera_name is None:
            return None
        return self.editable_cameras[active_camera_name]

    @property
    def gizmo_matrix(self):
        active_camera = self.active_camera
        if active_camera is None:
            return np.eye(4)
        return self._get_camera_global_pose(active_camera).to_transformation_matrix()

    @gizmo_matrix.setter
    def gizmo_matrix(self, matrix):
        active_camera = self.active_camera
        if active_camera is None:
            return
        self._set_camera_global_pose(active_camera, sapien.Pose(matrix))
        self.viewer.notify_render_update()

    def get_ui_windows(self):
        self._build_ui()
        if self._ui_window is None:
            return []
        return [self._ui_window]

    def after_render(self):
        if self.viewer.scene is None:
            return
        self._update_camera_linesets()

    def look_through_active_camera(self, _=None):
        active_camera = self.active_camera
        if active_camera is None:
            return
        self.viewer.set_camera_pose(self._get_camera_global_pose(active_camera))

    def set_active_camera_from_view(self, _=None):
        active_camera = self.active_camera
        if active_camera is None:
            return
        self._set_camera_global_pose(
            active_camera, self.viewer.window.get_camera_pose()
        )
        self.viewer.notify_render_update()

    def _build_ui(self):
        if self.viewer.scene is None or not self.camera_names:
            self._ui_window = None
            return

        if self._ui_window is None:
            self._gizmo = R.UIGizmo().Bind(self, "gizmo_matrix")
            self._camera_selector = (
                R.UIOptions()
                .Label("Camera")
                .Style("select")
                .BindItems(self, "camera_items")
                .BindIndex(self, "active_camera_index")
            )

            self._ui_window = (
                R.UIWindow()
                .Label("Camera Editor")
                .Pos(10, 420)
                .Size(420, 260)
                .append(
                    R.UICheckbox().Label("Enabled").Bind(self, "enabled"),
                    self._camera_selector,
                    R.UIDisplayText().Text(
                        "Click a camera frustum, then drag the gizmo to move it."
                    ),
                    R.UISameLine().append(
                        R.UIButton()
                        .Label("Look Through")
                        .Callback(self.look_through_active_camera),
                        R.UIButton()
                        .Label("Set From View")
                        .Callback(self.set_active_camera_from_view),
                    ),
                    R.UIConditional()
                    .Bind(lambda: self.enabled and self.active_camera is not None)
                    .append(self._gizmo),
                )
            )

        projection = self.viewer.window.get_camera_projection_matrix()
        view = (
            (self.viewer.window.get_camera_pose() * _VIEWER_GL_POSE)
            .inv()
            .to_transformation_matrix()
        )
        self._gizmo.CameraMatrices(view, projection)
        self._gizmo.Matrix(self.gizmo_matrix)

    def _create_visual_models(self):
        self._line_set_model = self.viewer.renderer_context.create_line_set(
            _CAMERA_LINESET_VERTICES,
            _INACTIVE_CAMERA_LINESET_COLORS,
        )
        self._active_line_set_model = self.viewer.renderer_context.create_line_set(
            _CAMERA_LINESET_VERTICES,
            _ACTIVE_CAMERA_LINESET_COLORS,
        )

    def _clear_camera_linesets(self):
        if self.viewer.render_scene is None:
            self._line_sets = []
            self._line_set_names = []
            self._line_set_active_name = None
            return
        for node in self._line_sets:
            self.viewer.render_scene.remove_node(node)
        self._line_sets = []
        self._line_set_names = []
        self._line_set_active_name = None

    def _update_camera_linesets(self):
        if not self.enabled or self.viewer.render_scene is None:
            self._clear_camera_linesets()
            return

        camera_names = self.camera_names
        if (
            len(self._line_sets) != len(camera_names)
            or self._line_set_names != camera_names
            or self._line_set_active_name != self.active_camera_name
        ):
            self._clear_camera_linesets()
            for camera_name in camera_names:
                line_set_model = (
                    self._active_line_set_model
                    if camera_name == self.active_camera_name
                    else self._line_set_model
                )
                self._line_sets.append(
                    self.viewer.render_scene.add_line_set(line_set_model)
                )
            self._line_set_names = list(camera_names)
            self._line_set_active_name = self.active_camera_name

        for line_set, camera_name in zip(self._line_sets, camera_names):
            camera = self.editable_cameras[camera_name]
            model_matrix = self._get_camera_model_matrix(camera)
            line_set.set_position(model_matrix[:3, 3])
            line_set.set_rotation(mat2quat(model_matrix[:3, :3]))
            line_set.set_scale(
                np.array(
                    [
                        np.tan(camera.camera.fovx / 2),
                        np.tan(camera.camera.fovy / 2),
                        1.0,
                    ]
                )
                * 0.3
            )

    def _handle_click(self, _viewer, x: int, y: int) -> bool:
        if not self.enabled:
            return False

        click_target = np.array([x, y], dtype=np.float32)
        best_match = None
        for camera_name, camera in self.editable_cameras.items():
            projected = self._project_camera_origin(camera)
            if projected is None:
                continue
            distance = np.linalg.norm(projected[:2] - click_target)
            if distance > self.selection_radius_px:
                continue
            candidate = (distance, projected[2], camera_name)
            if best_match is None or candidate < best_match:
                best_match = candidate

        if best_match is None:
            return False

        self.active_camera_name = best_match[2]
        self.viewer.select_entity(None)
        return True

    def _project_camera_origin(self, camera: Camera):
        segmentation_width, segmentation_height = self.viewer.window.get_picture_size(
            "Segmentation"
        )
        pose = self._get_camera_global_pose(camera)
        point = np.array([pose.p[0], pose.p[1], pose.p[2], 1.0], dtype=np.float32)
        view = (
            (self.viewer.window.get_camera_pose() * _VIEWER_GL_POSE)
            .inv()
            .to_transformation_matrix()
        )
        clip = self.viewer.window.get_camera_projection_matrix() @ (view @ point)
        if clip[3] <= 1e-6:
            return None
        ndc = clip[:3] / clip[3]
        if ndc[2] < -1 or ndc[2] > 1:
            return None
        px = (ndc[0] * 0.5 + 0.5) * segmentation_width
        py = (1 - (ndc[1] * 0.5 + 0.5)) * segmentation_height
        return np.array([px, py, ndc[2]], dtype=np.float32)

    def _get_camera_global_pose(self, camera: Camera) -> sapien.Pose:
        pose = camera.camera.get_global_pose()
        if len(pose) > 1:
            pose = pose[0]
        return pose.sp

    def _get_camera_model_matrix(self, camera: Camera) -> np.ndarray:
        matrix = camera.camera.get_model_matrix()
        if hasattr(matrix, "detach"):
            matrix = matrix.detach().cpu().numpy()
        else:
            matrix = np.asarray(matrix)
        if matrix.ndim == 3:
            matrix = matrix[0]
        return matrix

    def _set_camera_global_pose(self, camera: Camera, pose: sapien.Pose):
        local_pose = pose
        if camera.camera.mount is not None:
            mount_pose = camera.camera.mount.pose
            if len(mount_pose) > 1:
                mount_pose = mount_pose[0]
            local_pose = mount_pose.sp.inv() * pose
        camera.camera.set_local_pose(local_pose)
        camera.config.pose = Pose.create(local_pose)
