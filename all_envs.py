import gym

def box_filter(env):
    if type(env.action_space) == gym.spaces.box.Box:
        return True
    if type(env.action_space) == tuple:
        return True
    return False


allenv = list(gym.envs.registry.all())

check = box_filter

print( "{} environments".format( len(allenv) ) )
print( "{}\t{}\t{}".format( "Env ID", "Obs Space", "Action Space" ) )
for envspec in allenv:
    try:
        env = envspec.make()
        if check is not None and check(env):
            print( "{}\t{}\t{}".format( 
                envspec.id,
                env.observation_space,
                env.action_space) )
    except Exception as ex:
        if check is None:
            print( "{}\tFailed: {}".format( envspec.id, ex ) )

