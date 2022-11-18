# TODO fix colors
dataset_info = dict(
    dataset_name="hockey",
    paper_info=dict(
        author="None",
        title="Hockey Dataset",
        container="None",
        year="2022",
        homepage="none",
    ),
    keypoint_info={
        0: dict(name="head", id=0, color=[51, 153, 255], type="upper", swap=""),
        1: dict(name="neck", id=1, color=[51, 153, 255], type="upper", swap=""),
        2: dict(
            name="right_shoulder",
            id=2,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        3:
        dict(
            name='right_elbow',
            id=3,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        4:
        dict(
            name='right_wrist',
            id=4,
            color=[255, 128, 0],
            type="upper",
            swap="left_wrist",
        ),
        5: dict(
            name="left_shoulder",
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        6:
        dict(
            name='left_elbow',
            id=6,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        7:
        dict(
            name='left_wrist',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        8:
        dict(
            name='right_hip',
            id=8,
            ##Change this
            color=[255, 128, 0],
            type='lower',
            swap='left_hip'),
        9:
        dict(
            name='right_knee',
            id=9,
            color=[255, 128, 0],
            type='lower',
            swap='left_knee'),
        10:
        dict(
            name='right_ankle',
            id=10,
            color=[255, 128, 0],
            type="lower",
            swap="left_ankle",
        ),
        11: dict(
            name="left_hip", id=11, color=[0, 255, 0], type="lower", swap="right_hip"
        ),
        12: dict(
            name="left_knee",
            id=12,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        13:
        dict(
            name='left_ankle',
            id=13,
            color=[0, 255, 0],
            type='lower',
            swap='right_ankle'),
    },
    skeleton_info={
        0:
        dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('right_ankle', 'right_knee'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('right_knee', 'right_hip'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('left_hip', 'right_hip'), id=4, color=[51, 153, 255]),
        5:
        dict(link=('left_shoulder', 'left_hip'), id=5, color=[0, 255, 0]),
        6:
        dict(link=('right_shoulder', 'right_hip'), id=6, color=[255, 128, 0]),
        7:
        dict(
            link=('left_shoulder', 'right_shoulder'),
            id=7,
            color=[51, 153, 255]),
        8:
        dict(link=('left_shoulder', 'left_elbow'), id=8, color=[0, 255, 0]),
        9:
        dict(
            link=('right_shoulder', 'right_elbow'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('right_elbow', 'right_wrist'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('neck', 'left_shoulder'), id=12, color=[51, 153, 255]),
        13:
        dict(link=('neck', 'right_shoulder'), id=13, color=[51, 153, 255]),
        14:
        dict(link=('neck', 'head'), id=14, color=[51, 153, 255]),
    },
    # TODO Tune this
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    ],
    # TODO Tune this
    sigmas=[
        0.026, 0.026, 0.0079, 0.072, 0.062, 0.0079, 0.072, 0.062, 0.107, 0.087,
        0.089, 0.107, 0.087, 0.089
    ])
