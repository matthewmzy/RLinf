# -*- coding: utf-8 -*-
"""RobotControl protobuf messages vendored from the A2D runtime."""

from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder

_sym_db = _symbol_database.Default()

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x13robot_control.proto\x12\rrobot_control"\x92\x04\n\x0bObservation\x12\x36\n\x06images\x18\x01 \x03(\x0b\x32&.robot_control.Observation.ImagesEntry\x12\x36\n\x06states\x18\x02 \x03(\x0b\x32&.robot_control.Observation.StatesEntry\x12>\n\ntimestamps\x18\x03 \x03(\x0b\x32*.robot_control.Observation.TimestampsEntry\x12\x19\n\x0c\x63ontrol_mode\x18\x04 \x01(\x05H\x00\x88\x01\x01\x12\x1d\n\x10trajectory_label\x18\x05 \x01(\x05H\x01\x88\x01\x01\x12\x1b\n\x0eis_switch_mode\x18\x06 \x01(\x08H\x02\x88\x01\x01\x1aG\n\x0bImagesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\'\n\x05value\x18\x02 \x01(\x0b\x32\x18.robot_control.ImageData:\x02\x38\x01\x1aG\n\x0bStatesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\'\n\x05value\x18\x02 \x01(\x0b\x32\x18.robot_control.ArrayData:\x02\x38\x01\x1a\x31\n\x0fTimestampsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x01:\x02\x38\x01\x42\x0f\n\r_control_modeB\x13\n\x11_trajectory_labelB\x11\n\x0f_is_switch_mode"Y\n\tImageData\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\x12\x0e\n\x06height\x18\x02 \x01(\x05\x12\r\n\x05width\x18\x03 \x01(\x05\x12\x10\n\x08\x63hannels\x18\x04 \x01(\x05\x12\r\n\x05\x64type\x18\x05 \x01(\t"*\n\tArrayData\x12\x0e\n\x06values\x18\x01 \x03(\x02\x12\r\n\x05shape\x18\x02 \x03(\x05"+\n\x06\x41\x63tion\x12\x0e\n\x06values\x18\x01 \x03(\x02\x12\x11\n\tdimension\x18\x02 \x01(\x05"*\n\x0cResetRequest\x12\x11\n\x04seed\x18\x01 \x01(\x05H\x00\x88\x01\x01\x42\x07\n\x05_seed"b\n\rResetResponse\x12/\n\x0bobservation\x18\x01 \x01(\x0b\x32\x1a.robot_control.Observation\x12\x0f\n\x07success\x18\x02 \x01(\x08\x12\x0f\n\x07message\x18\x03 \x01(\t"4\n\x0bStepRequest\x12%\n\x06\x61\x63tion\x18\x01 \x01(\x0b\x32\x15.robot_control.Action"a\n\x0cStepResponse\x12/\n\x0bobservation\x18\x01 \x01(\x0b\x32\x1a.robot_control.Observation\x12\x0f\n\x07success\x18\x02 \x01(\x08\x12\x0f\n\x07message\x18\x03 \x01(\t"\x0f\n\rHealthRequest"2\n\x0eHealthResponse\x12\x10\n\x08is_ready\x18\x01 \x01(\x08\x12\x0e\n\x06status\x18\x02 \x01(\t"\x0f\n\rGetObsRequest"c\n\x0eGetObsResponse\x12/\n\x0bobservation\x18\x01 \x01(\x0b\x32\x1a.robot_control.Observation\x12\x0f\n\x07success\x18\x02 \x01(\x08\x12\x0f\n\x07message\x18\x03 \x01(\t"9\n\x10SetActionRequest\x12%\n\x06\x61\x63tion\x18\x01 \x01(\x0b\x32\x15.robot_control.Action"5\n\x11SetActionResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\x12\x0f\n\x07message\x18\x02 \x01(\t2\xf9\x02\n\x0cRobotControl\x12\x42\n\x05reset\x12\x1b.robot_control.ResetRequest\x1a\x1c.robot_control.ResetResponse\x12?\n\x04step\x12\x1a.robot_control.StepRequest\x1a\x1b.robot_control.StepResponse\x12\x46\n\x07get_obs\x12\x1c.robot_control.GetObsRequest\x1a\x1d.robot_control.GetObsResponse\x12O\n\nset_action\x12\x1f.robot_control.SetActionRequest\x1a .robot_control.SetActionResponse\x12K\n\x0chealth_check\x12\x1c.robot_control.HealthRequest\x1a\x1d.robot_control.HealthResponseb\x06proto3'
)

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, "robot_control_pb2", _globals)
if not _descriptor._USE_C_DESCRIPTORS:
    DESCRIPTOR._loaded_options = None
    _globals["_OBSERVATION_IMAGESENTRY"]._loaded_options = None
    _globals["_OBSERVATION_IMAGESENTRY"]._serialized_options = b"8\001"
    _globals["_OBSERVATION_STATESENTRY"]._loaded_options = None
    _globals["_OBSERVATION_STATESENTRY"]._serialized_options = b"8\001"
    _globals["_OBSERVATION_TIMESTAMPSENTRY"]._loaded_options = None
    _globals["_OBSERVATION_TIMESTAMPSENTRY"]._serialized_options = b"8\001"
