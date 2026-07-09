# Launch file for Insta360 ROS driver
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ld = LaunchDescription()

    #=============== Launch Arguments ===============#

    equirectangular = LaunchConfiguration("equirectangular")
    ld.add_action(DeclareLaunchArgument("equirectangular", default_value="false", choices=["true", "false"]))

    imu_filter = LaunchConfiguration("imu_filter")
    ld.add_action(DeclareLaunchArgument("imu_filter", default_value="true", choices=["true", "false"]))



    #=============== Config ===============#

    equirectangular_config = LaunchConfiguration("equirectangular_config")
    ld.add_action(
        DeclareLaunchArgument(
            "equirectangular_config",
            default_value=PathJoinSubstitution(
                [FindPackageShare("insta360_ros_driver"), "config", "equirectangular.yaml"]
            ),
        )
    )

    imu_config = LaunchConfiguration("imu_config")
    ld.add_action(
        DeclareLaunchArgument(
            "imu_config",
            default_value=PathJoinSubstitution(
                [FindPackageShare("insta360_ros_driver"), "config", "imu_filter.yaml"]
            ),
        )
    )



    #=============== Nodes ===============#

    driver_node = Node(
        package="insta360_ros_driver",
        executable="insta360_ros_driver",
        name="insta360_ros_driver",
        output="log",
    )
    ld.add_action(driver_node)

    decoder_node = Node(
        package="insta360_ros_driver",
        executable="decoder",
        name="image_decoder",
        parameters=[
            {
                "compressed_topic": "/dual_fisheye/image/compressed",
                "uncompressed_topic": "/dual_fisheye/image",
                "skip_frame": 0,
                "i_frame_only": False,
            }
        ],
        output="log",
    )
    ld.add_action(decoder_node)

    equirectangular_node = Node(
        package="insta360_ros_driver",
        executable="equirectangular_cpp",
        name="equirectangular_node",
        parameters=[equirectangular_config],
        condition=IfCondition(equirectangular),
        output="screen",
    )
    ld.add_action(equirectangular_node)

    imu_filter_node = Node(
        package="imu_filter_madgwick",
        executable="imu_filter_madgwick_node",
        name="imu_filter",
        parameters=[imu_config],
        condition=IfCondition(imu_filter),
        output="log",
    )
    ld.add_action(imu_filter_node)

    return ld