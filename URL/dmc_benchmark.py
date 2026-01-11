DOMAINS = [
    'walker',
    'quadruped',
    'jaco',
]

WALKER_TASKS = [
    'walker_stand',
    'walker_walk',
    'walker_run',
    'walker_flip',
]

QUADRUPED_TASKS = [
    'quadruped_walk',
    'quadruped_run',
    'quadruped_stand',
    'quadruped_jump',
]

JACO_TASKS = [
    'jaco_reach_top_left',
    'jaco_reach_top_right',
    'jaco_reach_bottom_left',
    'jaco_reach_bottom_right',
]

TASKS = WALKER_TASKS + QUADRUPED_TASKS + JACO_TASKS

parameter_1 = ['0.4', '0.8', '1.0', '1.4']
parameter_1_eval = ['0.6', '1.2']

parameter_2 = ['0.2', '0.6', '1.0', '1.4', '1.8']
parameter_2_eval = ['0.4', '0.8', '1.2', '1.6']

PRETRAIN_TASKS = {
    'walker': 'walker_stand',
    'jaco': 'jaco_reach_top_left',
    'quadruped': 'quadruped_walk',
    'walker_mass': ['walker_stand~mass~' + para for para in parameter_1],
    'quadruped_mass': ['quadruped_stand~mass~' + para for para in parameter_1],
    'jaco_mass': ['jaco_reach_top_left~mass~' + para for para in parameter_1],
    'walker_mass2': ['walker_stand~mass~' + para for para in parameter_2],
    'quadruped_mass2': ['quadruped_stand~mass~' + para for para in parameter_2],
    'quadruped_damping3': ['quadruped_stand~damping~' + para for para in parameter_2],
    'jaco_mass2': ['jaco_reach_top_left~mass~' + para for para in parameter_2],
    'jaco_damping3': ['jaco_reach_top_left~damping~' + para for para in parameter_2],
}

FINETUNE_TASKS = {
    'walker_stand_mass': ['walker_stand~mass~' + para for para in parameter_1],
    'walker_stand_mass_eval': ['walker_stand~mass~' + para for para in parameter_1_eval],
    'walker_walk_mass': ['walker_walk~mass~' + para for para in parameter_1],
    'walker_walk_mass_eval': ['walker_walk~mass~' + para for para in parameter_1_eval],
    'walker_run_mass': ['walker_run~mass~' + para for para in parameter_1],
    'walker_run_mass_eval': ['walker_run~mass~' + para for para in parameter_1_eval],
    'walker_flip_mass': ['walker_flip~mass~' + para for para in parameter_1],
    'walker_flip_mass_eval': ['walker_flip~mass~' + para for para in parameter_1_eval],

    'walker_stand_mass2': ['walker_stand~mass~' + para for para in parameter_2],
    'walker_stand_mass2_eval': ['walker_stand~mass~' + para for para in parameter_2_eval],
    'walker_walk_mass2': ['walker_walk~mass~' + para for para in parameter_2],
    'walker_walk_mass2_eval': ['walker_walk~mass~' + para for para in parameter_2_eval],
    'walker_run_mass2': ['walker_run~mass~' + para for para in parameter_2],
    'walker_run_mass2_eval': ['walker_run~mass~' + para for para in parameter_2_eval],
    'walker_flip_mass2': ['walker_flip~mass~' + para for para in parameter_2],
    'walker_flip_mass2_eval': ['walker_flip~mass~' + para for para in parameter_2_eval],

    'quadruped_stand_mass': ['quadruped_stand~mass~' + para for para in parameter_1],
    'quadruped_stand_mass_eval': ['quadruped_stand~mass~' + para for para in parameter_1_eval],
    'quadruped_walk_mass': ['quadruped_walk~mass~' + para for para in parameter_1],
    'quadruped_walk_mass_eval': ['quadruped_walk~mass~' + para for para in parameter_1_eval],
    'quadruped_run_mass': ['quadruped_run~mass~' + para for para in parameter_1],
    'quadruped_run_mass_eval': ['quadruped_run~mass~' + para for para in parameter_1_eval],
    'quadruped_jump_mass': ['quadruped_jump~mass~' + para for para in parameter_1],
    'quadruped_jump_mass_eval': ['quadruped_jump~mass~' + para for para in parameter_1_eval],

    'quadruped_stand_mass2': ['quadruped_stand~mass~' + para for para in parameter_2],
    'quadruped_stand_mass2_eval': ['quadruped_stand~mass~' + para for para in parameter_2_eval],
    'quadruped_walk_mass2': ['quadruped_walk~mass~' + para for para in parameter_2],
    'quadruped_walk_mass2_eval': ['quadruped_walk~mass~' + para for para in parameter_2_eval],
    'quadruped_run_mass2': ['quadruped_run~mass~' + para for para in parameter_2],
    'quadruped_run_mass2_eval': ['quadruped_run~mass~' + para for para in parameter_2_eval],
    'quadruped_jump_mass2': ['quadruped_jump~mass~' + para for para in parameter_2],
    'quadruped_jump_mass2_eval': ['quadruped_jump~mass~' + para for para in parameter_2_eval],

    'quadruped_stand_damping3': ['quadruped_stand~damping~' + para for para in parameter_2],
    'quadruped_stand_damping3_eval': ['quadruped_stand~damping~' + para for para in parameter_2_eval],
    'quadruped_walk_damping3': ['quadruped_walk~damping~' + para for para in parameter_2],
    'quadruped_walk_damping3_eval': ['quadruped_walk~damping~' + para for para in parameter_2_eval],
    'quadruped_run_damping3': ['quadruped_run~damping~' + para for para in parameter_2],
    'quadruped_run_damping3_eval': ['quadruped_run~damping~' + para for para in parameter_2_eval],
    'quadruped_jump_damping3': ['quadruped_jump~damping~' + para for para in parameter_2],
    'quadruped_jump_damping3_eval': ['quadruped_jump~damping~' + para for para in parameter_2_eval],

    'jaco_tl_mass': ['jaco_reach_top_left~mass~' + para for para in parameter_1],
    'jaco_tl_mass_eval': ['jaco_reach_top_left~mass~' + para for para in parameter_1_eval],
    'jaco_tr_mass': ['jaco_reach_top_right~mass~' + para for para in parameter_1],
    'jaco_tr_mass_eval': ['jaco_reach_top_right~mass~' + para for para in parameter_1_eval],
    'jaco_bl_mass': ['jaco_reach_bottom_left~mass~' + para for para in parameter_1],
    'jaco_bl_mass_eval': ['jaco_reach_bottom_left~mass~' + para for para in parameter_1_eval],
    'jaco_br_mass': ['jaco_reach_bottom_right~mass~' + para for para in parameter_1],
    'jaco_br_mass_eval': ['jaco_reach_bottom_right~mass~' + para for para in parameter_1_eval],

    'jaco_tl_mass2': ['jaco_reach_top_left~mass~' + para for para in parameter_2],
    'jaco_tl_mass2_eval': ['jaco_reach_top_left~mass~' + para for para in parameter_2_eval],
    'jaco_tr_mass2': ['jaco_reach_top_right~mass~' + para for para in parameter_2],
    'jaco_tr_mass2_eval': ['jaco_reach_top_right~mass~' + para for para in parameter_2_eval],
    'jaco_bl_mass2': ['jaco_reach_bottom_left~mass~' + para for para in parameter_2],
    'jaco_bl_mass2_eval': ['jaco_reach_bottom_left~mass~' + para for para in parameter_2_eval],
    'jaco_br_mass2': ['jaco_reach_bottom_right~mass~' + para for para in parameter_2],
    'jaco_br_mass2_eval': ['jaco_reach_bottom_right~mass~' + para for para in parameter_2_eval],

    'jaco_tl_damping3': ['jaco_reach_top_left~damping~' + para for para in parameter_2],
    'jaco_tl_damping3_eval': ['jaco_reach_top_left~damping~' + para for para in parameter_2_eval],
    'jaco_tr_damping3': ['jaco_reach_top_right~damping~' + para for para in parameter_2],
    'jaco_tr_damping3_eval': ['jaco_reach_top_right~damping~' + para for para in parameter_2_eval],
    'jaco_bl_damping3': ['jaco_reach_bottom_left~damping~' + para for para in parameter_2],
    'jaco_bl_damping3_eval': ['jaco_reach_bottom_left~damping~' + para for para in parameter_2_eval],
    'jaco_br_damping3': ['jaco_reach_bottom_right~damping~' + para for para in parameter_2],
    'jaco_br_damping3_eval': ['jaco_reach_bottom_right~damping~' + para for para in parameter_2_eval],
}
