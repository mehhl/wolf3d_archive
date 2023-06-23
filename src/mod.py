""" A script for producing Dog-Hide task, based on dm_control's Dog-Run. """

from dm_control.suite import dog
from dm_control import mujoco
from dm_control import composer
from dm_control import mjcf

import numpy as np
import os
import warnings
import collections


if hasattr(__builtins__, '__IPYTHON__'):
    _ASSET_DIR = os.path.join(os.getcwd(), '..', 'assets')
else:
    _ASSET_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets')
      
# OK
def get_model_and_agents(*, count_seekers=1, xsize=50, ysize=50): 
    """Wrapper for adding/modifying and reading assets required by Dog-Hide."""
    model = mjcf.RootElement()
    agents = {}
    hfield = model.asset.add('hfield', name='heightmap', 
                             file=os.path.join(_ASSET_DIR, 'heightmap.png'),
                             size=[xsize, ysize, .1, 1])
    hfloor = model.worldbody.add('geom', type='hfield', 
                                 name='floor', hfield=hfield) 
    learner = mjcf.from_path(os.path.join(_ASSET_DIR, 'dog.xml'))
    learner.model = 'learner'
    # get rid of unneeded stuff
    learner.compiler.remove()        # this removes previous asset path
    removals = []
    removals.append(learner.find('body', 'ball'))
    removals.append(learner.find('camera', 'ball'))
    learner.worldbody.body[0].all_children()[0].remove()
    removals.extend(learner.worldbody.geom)
    for element in removals:
        element.remove()
    spawn_pos = learner.find('body', 'torso').pos.copy()
    spawn_site = model.worldbody.add('site', pos=spawn_pos, group=3)
    model.attach(learner).add('freejoint')
    agents['learner'] = learner
    
    
    # add seekers
    seekers = [mjcf.from_path(os.path.join(_ASSET_DIR, 'quadruped.xml')) \
               for i in range(count_seekers)]
    for i, seeker in enumerate(seekers):
        seeker = mjcf.from_path(os.path.join(_ASSET_DIR, 'quadruped.xml'))
        # remove the unnecessary parts of the seeker
        removals = []
        removals.append(seeker.find('body', 'ball'))
        removals.extend(seeker.find_all('camera'))
        removals.extend(seeker.worldbody.geom)
        seeker.worldbody.body[0].all_children()[0].remove()
        for element in removals:
            element.remove()
        slight = seeker.find('body', 'torso').add('light',
                                                  pos=[0, 0, 0], dir=[0, 0, 1],
                                                  cutoff=10, exponent=50, 
                                                  attenuation=[1, 0, .01])
        seeker.model = 'Seeker' + str(i) 
        # attach the seekers at random locations
        r = (xsize + ysize)/3
        h = 1.2
        arg = np.random.uniform(0, 2*np.pi)
        xpos, ypos, zpos = r*np.cos(arg), r*np.sin(arg), h
        spawn_site = model.worldbody.add('site', 
                                         pos=[xpos, ypos, zpos], group=3)
        spawn_site.attach(seeker).add('freejoint')
        agents['Seeker' + str(i)] = seeker
        
    return model, agents

# OK
class _SeekerPhysics:
    """Physics methods that only affect the Seekers. Most methods here
    are ported from `quadruped.Physics` 
    (in `dm_control`'s Quadruped domain.)"""

    def __init__(self, physics: mujoco.Physics, seekers: list):
        """Initialize an instance of `SeekerPhysics`.

        Args:
            `physics` -- the instance of `mujoco.Physics` that this object
            is bound to.
            `seekers` -- a list of seeker IDs (eg ['Seeker1, 'Seeker2', ...])
            that will be used to identify seeker bodies 
            inside the composed model."""
        self.physics = physics
        self._d = physics.named.data
        self._m = physics.model
        self._s = seekers
 
        __jnt_dfadr = physics.named.model.jnt_dofadr
        __jnt_name = physics.named.model.jnt_dofadr.axes.row.names
        __get_dfadr = lambda s: __jnt_dfadr[np.char.startswith(__jnt_name, s)]
        self._s_jnt_dofadr = collections.OrderedDict(
                             (s, __get_dfadr(s)) for s in _s)

        __jnt_qadr = physics.named.model.jnt_qposadr
        __jnt_qname = physics.named.model.jnt_qposadr.axes.row.names
        __get_qadr = lambda s: __jnt_qadr[np.char.startswith(__jnt_qname, s)]
        self._s_jnt_qposadr = collections.OrderedDict(
                         (s, __get_qadr(s)) for s in _s)

        __act_name = physics.named.data.act.axes.row.names
        __get_act = lambda s: np.char.startswith(__act_name, s)
        self._s_act = collections.OrderedDict((s, __get_act(s)) for s in _s)

        __toes = ['toe_front_left',
                  'toe_back_left',
                  'toe_back_right',
                  'toe_front_right']
        __get_toes = lambda s: [s + '/' + toe for toe in __toes]
        self._s_toes = collections.OrderedDict((s, __get_toes(s)) for s in _s)

    def torso_upright(self):
        """Returns the dot-product of each seeker's torso z-axis 
        and the global z-axis."""
        entry = lambda s: self._d.xmat[s + '/torso', 'zz']
        return np.vstack([entry(s) for s in self._s])

    def torso_velocity(self):
        """Returns the velocity of each seeker's torso, 
        in their local frame."""
        entry = lambda s : self._d.sensordata[s + '/velocimeter']
        return np.vstack([entry(s) for s in self._s])

    def egocentric_state(self):
        """Returns local coordinates for each seeker's joints."""
        # TODO: Fix the '[1:]'; this is a workaround to remove
        # hinge joints that cant be easily filtered out in database manner
        hinge_jnt = self.physics.model.jnt_type == _HINGE_TYPE 
        entry = lambda s: np.hstack((self._d.qpos[self._s_jnt_qposadr[s][1:]],
                                     self._d.qvel[self._s_jnt_dofadr[s][1:]],
                                     self.physics.data.act[self._s_act[s]]))
        return np.vstack([entry(s) for s in self._s])

    def toe_positions(self):
        """Returns toe positions in egocentric frame of each Seeker."""
        
        torso_frame = lambda s: self._d.xmat[s + '/torso'].reshape(3, 3)
        torso_pos = lambda s: self._d.xpos[s + '/torso']
        torso_to_toe = lambda s: self._d.xpos[self._s_toes[s]] - torso_pos(s)
        return np.vstack([[torso_to_toe(s).dot(torso_frame(s)) 
                           for s in self._s]])

    def force_torque(self):
        """Returns scaled orce/torque sensor readings at each Seeker' toes."""
        [f_sensor_ids] = np.where(np.in1d(self._m.sensor_type,
                                          enums.mjtSensor.mjSENS_FORCE)) 
        [t_sensor_ids] = np.where(np.in1d(self._m.sensor_type,
                                          enums.mjtSensor.mjSENS_TORQUE))
        ft_sensor_ids = np.hstack([f_sensor_ids, t_sensor_ids])
        ft_sensor_names = [_m.id2name(sid, 'sensor') for sid in ft_sensor_ids]
        get_sensors = lambda s: [n for n in ft_sensor_names if n.startswith(s)]
        entry = lambda s: np.arcsinh(self._d.sensordata[get_sensors(s)])
        return np.vstack([entry(s) for s in self._s])
        
    def imu(self):
        """Return IMU-like sensor readings for each seeker."""
        [gyro_sensors] = np.where(np.in1d(self._m.sensor_type,
                                          enums.mjtSensor.mjSENS_GYRO))
        [acc_sensors] = np.where(np.in1d(self._m.sensor_type,
                                         enums.mjtSensor.mjSENS_ACCELEROMETER))
        imu_sensors = np.hstack([gyro_sensors, acc_sensors])
        imu_sensor_names = [_m.id2name(sid, 'sensor') for sid in imu_sensors]
        get_sensors = lambda s: [n for n in imu_sensor_names if n.startswith(s)]
        entry = lambda s: self._d.sensordata[get_sensors(s)]
        return np.vstack([entry(s) for s in self._s])

    def rangefinder(self):
        """Returns scaled rangefinder sensor readings for each seeker."""
        [rf_sensors] = np.where(np.in1d(self._m.sensor_type,
                                        enums.mjtSensor.mjSENS_RANGEFINDER))
        rf_sensor_names = [self._m.id2name(sid, 'sensor') for sid in rf_sensors]
        get_sensors = lambda s: [n for n in rf_sensor_names if n.startswith(s)]
        rf_readings = lambda s: self._d.sensordata[get_sensors(s)]
        no_intersection = -1.0
        entry = lambda s: np.where(rf_readings(s) == no_intersection,
                                   1.0,
                                   np.tanh(rf_readings(s)))
        return np.vstack([entry(s) for s in self._s])

    def origin(self):
        """Returns origin position in each seeker's torso frame."""
        torso_frame = lambda s: self._d.xmat[s + '/torso'].reshape(3, 3)
        torso_pos = lambda s: self._d.xpos[s + '/torso']
        return np.vstack([-torso_pos(s).dot(torso_frame(s)) for s in self._s])



# this entire class needs to be REFACTORED to not use PyMJCF's implementation details
# but for now it'll have to do
class Physics(mjcf.physics.Physics):
    """ Wrapper for specialized `mjcf.physics.Physics` class
        to support light and heat sensors for multi-agent learning.
        Includes additional methods for Dog domain from `dog.Physics`."""

    @classmethod
    def from_mjcf_model(cls, model, /, learner, seekers=None):
        """ Constructs a new `mjcf.Physics` from `model` and reads list of
            agents from `agents`.

        Args:
            `model` - a MJCF `dm_control.mjcf.element.RootElement` model.
            `learner` - a string containing learner's name in the model.
            `seekers` - an iterable containing seekers' names."""
        physics = super().from_mjcf_model(model)
        physics._learner = learner
        if seekers is not None:
            _seekers = [name for name in seekers]
            physics._seekers = _seekers
            physics._seekers_torso = [name + '/torso' for name in seekers]
            physics._seeker_physics = _SeekerPhysics(physics, _seekers)
        return physics

    def light_and_heat_obs(self):
        """Returns light and heat values as perceived at Learner's skull."""
        observation_point = self.named.data.xpos[self._learner + '/skull']
        pos_seekers = [self.named.data.xpos[s] for s in self._seekers_torso]
        seeker_distances = np.linalg.norm(observation_point - pos_seekers, 
                                          axis=1)
        sigma = 2
        observed_heat = sum(np.exp(-(seeker_distances**2) / (2* sigma**2)))

        u = observation_point - pos_seekers
        v = [self.named.data.light_xdir[s + '//unnamed_light_1'] 
             for s in self._seekers]
        u_dot_v = np.einsum('ij, ij->i', u, v)
        d = np.linalg.norm(u, axis=1)**(-2)
        observed_light = np.sum(np.clip(u_dot_v, 0, 1)*d)
        return np.vstack((observed_heat, observed_light))

    # TODO write a method to verify correctness of all named objects used
    def torso_pelvis_height(self):
        """Returns the height of the torso. 
           Ported from `dog.Physics`."""
        return self.named.data.xpos[['learner/torso', 'learner/pelvis'], 'z']

    def z_projection(self):
        """Returns projection from Learner's z-axes to the z-axis of world.
           Ported from `dog.Physics`."""
        return np.vstack((self.named.data.xmat['learner/skull', 
                                               ['zx', 'zy', 'zz']],
                          self.named.data.xmat['learner/torso', 
                                               ['zx', 'zy', 'zz']],
                          self.named.data.xmat['learner/pelvis', 
                                               ['zx', 'zy', 'zz']]))

    def upright(self):
        """Returns projection from Learner's z-axes to the z-axis of world.
           Ported from `dog.Physics`."""
        return self.z_projection()[:, 2]

    def center_of_mass_velocity(self):
        """Returns the velocity of the Learner's center-of-mass. 
           Ported from `dog.Physics`."""
        return self.named.data.sensordata['learner/torso_linvel']

    def torso_com_velocity(self):
        """Returns the velocity of the center-of-mass 
        in the Learner's torso frame.
           Ported from `dog.Physics`."""
        torso_frame = self.named.data.xmat['learner/torso'].reshape(3, 3).copy()
        return self.center_of_mass_velocity().dot(torso_frame)
           
    def com_forward_velocity(self):
        """Returns the com velocity in the Learner's torso's forward direction.
           Ported from `dog.Physics`."""
        return self.torso_com_velocity()[0]

    def joint_angles(self):
        """Returns the configuration of all hinge joints (skipping free joints).
           Ported from `dog.Physics`, includes only Learner's joints."""
        hinge_joints = self.model.jnt_type == _HINGE_TYPE 
        hinge_names = self.named.model.jnt_type.axes.row.names
        learner_joints = np.char.startswith(hinge_names, 'learner')
        hinge_learner_joints = hinge_joints * learner_joints
        qpos_index = self.model.jnt_qposadr[hinge_learner_joints]
        return self.data.qpos[qpos_index].copy()

    def joint_velocities(self):
        """Returns the velocity of all hinge joints (skipping free joints).
           Ported from `dog.Physics`, includes only Learner's joints"""
        hinge_joints = self.model.jnt_type == _HINGE_TYPE
        hinge_names = self.named.model.jnt_type.axes.row.names
        learner_joints = np.char.startswith(hinge_names, 'learner')
        hinge_learner_joints = hinge_joints * learner_joints
        qvel_index = self.model.jnt_dofadr[hinge_learner_joints]
        return self.data.qvel[qvel_index].copy()

    def inertial_sensors(self):
        """Returns inertial sensor readings of Learner.
        Ported from `dog.Physics`."""
        return self.named.data.sensordata[['learner/accelerometer', 
                                           'learner/velocimeter',
                                           'learner/gyro']]

    def touch_sensors(self):
        """Returns Learner's touch readings. Ported from `dog.Physics`."""
        return self.named.data.sensordata[['learner/palm_L',
                                           'learner/palm_R',
                                           'learner/sole_L',
                                           'learner/sole_R']]

    def foot_forces(self):
        """Returns Learner's touch readings. Ported from `dog.Physics`."""
        return self.named.data.sensordata[['learner/foot_L',
                                           'learner/foot_R',
                                           'learner/hand_L',
                                           'learner/hand_R']]
# OK
physics = Physics.from_mjcf_model(model, 'learner', seekers)



SUITE = dog.SUITE
_DEFAULT_TIME_LIMIT = dog._DEFAULT_TIME_LIMIT
@SUITE.add('no_reward_visualization')
def hide(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    # blablabla

class Hide(dog.Move):
    # add new reward

    def __init__(self, random=None, observe_reward_factors=False):
        super().__init__(random=random, observe_reward_factors=observe_reward_factors)

    def initialize_episode(self, physics):
        super().initialize_episode(self, physics)

        # Randomize seeker positions.
        physics.data.qvel[0] = self.random.randn()
        seeker_positions = [physics.named.data.xpos[name] for name in seeker_names]



# Task
from dm_control.utils import containers
from dm_control.suite import dog


@SUITE.add('no_reward_visualization')
def hide(time_limit=dog._DEFAULT_TIME_LIMIT

###############TESTS

# OK
physics.model.nu == physics.model.na

# Do simulation tests on new physics
