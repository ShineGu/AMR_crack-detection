#include <ros/ros.h>
#include <geometry_msgs/Twist.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vel_pub");
    ros::NodeHandle n;
    ros::Publisher vel_pub = n.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
    ros::Rate loop_rate(10);
    int count = 0;

    double velx[] = {0, 0.2, -0.2};
    double vely[] = {0, 0.2, -0.2};
    double angle[] = {0, 0.6, -0.6};

    while(ros::ok())
    {
        geometry_msgs::Twist vel_msg;

        if(count / 10 >= 0)
        {
            vel_msg.linear.x = velx[1];
            vel_msg.linear.y = vely[0];
            vel_msg.angular.z = angle[0];
        }

        if(count / 10 >= 4)
        {
            vel_msg.linear.x = velx[2];
            vel_msg.linear.y = vely[0];
            vel_msg.angular.z = angle[0];
        }

        if(count / 10 >= 8)
        {
            vel_msg.linear.x = velx[0];
            vel_msg.linear.y = vely[1];
            vel_msg.angular.z = angle[0];
        }

        if(count / 10 >= 12)
        {
            vel_msg.linear.x = velx[0];
            vel_msg.linear.y = vely[2];
            vel_msg.angular.z = angle[0];
        }

        if(count / 10 >= 16)
        {
            vel_msg.linear.x = velx[0];
            vel_msg.linear.y = vely[0];
            vel_msg.angular.z = angle[1];
        }

        if(count / 10 >= 27)
        {
            vel_msg.linear.x = velx[0];
            vel_msg.linear.y = vely[0];
            vel_msg.angular.z = angle[2];
        }

        if(count / 10 >= 38)
        {
            vel_msg.linear.x = velx[0];
            vel_msg.linear.y = vely[0];
            vel_msg.angular.z = angle[0];
        }


        vel_pub.publish(vel_msg);
        ROS_INFO("Publish velocity command[%0.2f m/s, %0.2f m/s, %0.2f rad/s]", vel_msg.linear.x, vel_msg.linear.y, vel_msg.angular.z);
        
        count ++;

        loop_rate.sleep();
    }

    return 0;
}