/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

/**
 * BrownianMotionPoseIDL
 *
 */
service BrownianMotionPoseIDL
{
    bool enable();
    bool disable();

    bool set_q_x(1:double q);
    bool set_q_y(1:double q);
    bool set_q_z(1:double q);
    bool set_q_yaw(1:double q);
    bool set_q_pitch(1:double q);
    bool set_q_roll(1:double q);

    string get_q_x();
    string get_q_y();
    string get_q_z();
    string get_q_yaw();
    string get_q_pitch();
    string get_q_roll();
}
