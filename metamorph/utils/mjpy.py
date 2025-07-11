import itertools

from lxml import etree
import mujoco_py
from typing import Dict, List
from mujoco_py import MjSim
from mujoco_py import load_model_from_xml


def mj_name2id(sim, type_, name):
    """Returns the mujoco id corresponding to name."""
    if type_ == "site":
        return sim.model.site_name2id(name)
    elif type_ == "geom":
        return sim.model.geom_name2id(name)
    elif type_ == "body":
        return sim.model.body_name2id(name)
    elif type_ == "sensor":
        return sim.model.sensor_name2id(name)
    else:
        raise ValueError("type_ {} is not supported.".format(type_))


def mj_id2name(sim, type_, id_):
    """Returns the mujoco name corresponding to id."""
    if type_ == "site":
        return sim.model.site_id2name(id_)
    elif type_ == "geom":
        return sim.model.geom_id2name(id_)
    elif type_ == "body":
        return sim.model.body_id2name(id_)
    elif type_ == "sensor":
        return sim.model.sensor_id2name(id_)
    else:
        raise ValueError("type_ {} is not supported.".format(type_))


def mjsim_from_etree(root):
    """Return MjSim from etree root."""
    return MjSim(mjmodel_from_etree(root))


def mjmodel_from_etree(root):
    """Return MjModel from etree root."""
    model_string = etree.tostring(root, encoding="unicode", pretty_print=True)
    return load_model_from_xml(model_string)


def joint_qpos_idxs(sim, joint_name):
    """Gets indexes for the specified joint's qpos values."""
    addr = sim.model.get_joint_qpos_addr(joint_name)
    if isinstance(addr, tuple):
        return list(range(addr[0], addr[1]))
    else:
        return [addr]


def qpos_idxs_from_joint_prefix(sim, prefix):
    """Gets indexes for the qpos values of all joints matching the prefix."""
    qpos_idxs_list = [
        joint_qpos_idxs(sim, name)
        for name in sim.model.joint_names
        if name.startswith(prefix)
    ]
    return list(itertools.chain.from_iterable(qpos_idxs_list))


def qpos_idxs_for_agent(sim):
    """Gets indexes for the qpos values of all agent joints."""
    agent_joints = names_from_prefixes(sim, ["root", "torso", "limb"], "joint")
    qpos_idxs_list = [joint_qpos_idxs(sim, name) for name in agent_joints]
    return list(itertools.chain.from_iterable(qpos_idxs_list))


def joint_qvel_idxs(sim, joint_name):
    """Gets indexes for the specified joint's qvel values."""
    addr = sim.model.get_joint_qvel_addr(joint_name)
    if isinstance(addr, tuple):
        return list(range(addr[0], addr[1]))
    else:
        return [addr]


def qvel_idxs_from_joint_prefix(sim, prefix):
    """Gets indexes for the qvel values of all joints matching the prefix."""
    qvel_idxs_list = [
        joint_qvel_idxs(sim, name)
        for name in sim.model.joint_names
        if name.startswith(prefix)
    ]
    return list(itertools.chain.from_iterable(qvel_idxs_list))


def qvel_idxs_for_agent(sim):
    """Gets indexes for the qvel values of all agent joints."""
    agent_joints = names_from_prefixes(sim, ["root", "torso", "limb"], "joint")
    qvel_idxs_list = [joint_qvel_idxs(sim, name) for name in agent_joints]
    return list(itertools.chain.from_iterable(qvel_idxs_list))


def geom_idxs_for_agent(sim):
    """Gets indexes for agent geoms."""
    agent_geoms = names_from_prefixes(sim, ["torso", "limb"], "geom")
    geom_idx_list = [
        mj_name2id(sim, "geom", geom_name) for geom_name in agent_geoms
    ]
    return geom_idx_list


def body_idxs_for_agent(sim):
    """Gets indexes for agent body."""
    agent_bodies = names_from_prefixes(sim, ["torso", "limb"], "body")
    body_idx_list = [
        mj_name2id(sim, "body", body_name) for body_name in agent_bodies
    ]
    return body_idx_list


def names_from_prefixes(sim, prefixes, elem_type):
    """Get all names of elem_type elems which match any of the prefixes."""
    all_names = getattr(sim.model, "{}_names".format(elem_type))
    matches = []
    for name in all_names:
        for prefix in prefixes:
            if name.startswith(prefix):
                matches.append(name)
                break
    return matches


def get_active_contacts(sim):
    num_contacts = sim.data.ncon
    contacts = sim.data.contact[:num_contacts]
    contact_geoms = [
        tuple(
            sorted(
                (
                    mj_id2name(sim, "geom", contact.geom1),
                    mj_id2name(sim, "geom", contact.geom2),
                )
            )
        )
        for contact in contacts
    ]
    return sorted(list(set(contact_geoms)))

JOINT_TYPES = {
    0: 'free',
    1: 'ball',
    2: 'slide',
    3: 'hinge'
}

def print_kinematic_tree(model):
    print(f"--- Kinematic Tree for Model ---")
    
    # 1. 创建一个从父物体ID到子物体ID列表的映射，方便查找
    child_map: Dict[int, List[int]] = {i: [] for i in range(model.nbody)}
    for i in range(1, model.nbody):  # 从1开始，因为body 0是world
        parent_id = model.body_parentid[i]
        child_map[parent_id].append(i)

    joint_to_actuators: Dict[int, List[int]] = {i: [] for i in range(model.njnt)}
    for i in range(model.nu):
        if model.actuator_trntype[i] == 0:
            joint_id = model.actuator_trnid[i, 0]
            if joint_id != -1:
                joint_to_actuators[joint_id].append(i)

    def _recursive_print(body_id: int, indent_level: int):
        prefix = "  " * indent_level
        body_name = model.body_id2name(body_id)
        
        print(f"{prefix}└── BODY: {body_name} (ID: {body_id})")

        num_jnt = model.body_jntnum[body_id]
        if num_jnt > 0:
            joint_start_addr = model.body_jntadr[body_id]
            for i in range(num_jnt):
                joint_id = joint_start_addr + i
                joint_name = model.joint_id2name(joint_id)
                joint_type_id = model.jnt_type[joint_id]
                joint_type_name = JOINT_TYPES.get(joint_type_id, "unknown")
                
                print(f"{prefix}    ├── JOINT: {joint_name} (ID: {joint_id}, Type: {joint_type_name})")

                actuator_ids = joint_to_actuators.get(joint_id, [])
                for act_id in actuator_ids:
                    actuator_name = model.actuator_id2name(act_id)
                    print(f"{prefix}    │   └── ACTUATOR: {actuator_name} (ID: {act_id})")

        children_ids = child_map.get(body_id, [])
        for child_id in children_ids:
            _recursive_print(child_id, indent_level + 1)

    _recursive_print(0, 0)
    print("--- End of Kinematic Tree ---")
