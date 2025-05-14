
mesocosm_data = pd.read_excel("./data/Mesocosmdata.xls", sheet_name=0)

key = jax.random.PRNGKey(2468)  

dentNoPara = mesocosm_data.iloc[:100][['rep', 'day', 'dent.adult']]

dentNoPara['day'] = (dentNoPara['day'] - 1) * 5 + 7

dentNoPara = dentNoPara.iloc[::-1].reset_index(drop=True)

data = []
dentadult = []
trails = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

for trail in trails:
    subset_data = dentNoPara[dentNoPara['rep'] == trail][['day', 'dent.adult']]
    subset_dentadult = dentNoPara[dentNoPara['rep'] == trail][['dent.adult']]
    data.append(subset_data)
    dentadult.append(subset_dentadult)

def transform_thetas(sigSn, sigF, f_Sn, rn, k_Sn, sigJn, theta_Sn, theta_Jn, lambda_Jn):
    return jnp.array([jnp.log(sigSn), jnp.log(sigF), jnp.log(f_Sn), jnp.log(rn), jnp.log(k_Sn), jnp.log(sigJn), jnp.log(theta_Sn), jnp.log(theta_Jn), jnp.log(lambda_Jn)])

def get_thetas(thetas):
    sigSn = jnp.exp(thetas[0])
    sigF = jnp.exp(thetas[1])
    f_Sn = jnp.exp(thetas[2])
    rn = jnp.exp(thetas[3])
    k_Sn = jnp.exp(thetas[4])
    sigJn = jnp.exp(thetas[5])
    theta_Sn = jnp.exp(thetas[6])
    theta_Jn = jnp.exp(thetas[7])
    lambda_Jn = jnp.exp(thetas[8])

    return sigSn, sigF, f_Sn, rn, k_Sn, sigJn, theta_Sn, theta_Jn, lambda_Jn
#state : "Sn","Jn" ,"error_count", "F", "T_Sn", "day_index"
#thetas :  sigSn, sigF, f_Sn, rn, k_Sn, sigJn, theta_Sn, theta_Jn, lambda_Jn

def rproc_loop_noi_rm(state, thetas, key, num_steps):
    # extract states and thetas
    Sn, Jn, F, T_Sn, error_count = state[0], state[1], state[2], state[3], state[4]
    sigSn, sigF, f_Sn, rn, k_Sn, sigJn, theta_Sn, theta_Jn, lambda_Jn = get_thetas(thetas)

    # contant
    delta = 0.013
    dt = 0.25
    
    def loop_body(i, loop_state):
        Sn, Jn, F, T_Sn, error_count, key = loop_state
        # progressing the states and updating them
        Sn_term = lambda_Jn * Jn * dt - theta_Sn * Sn * dt - delta * Sn * dt
        Jn_term = rn * f_Sn * F * Sn * dt - lambda_Jn * Jn * dt - theta_Jn * Jn * dt - delta * Jn * dt
        F_term = -f_Sn * F * (Sn + 1 * Jn) * dt - delta * F * dt + 0.37 * dt

        # keep updating the states
        Sn = Sn_term + Sn
        Jn = Jn_term + Jn
        F = F_term + F

        # extreme cases
        Sn = jnp.where((Sn < 0.0) | (Sn > 1e5), 0.0, Sn)
        error_count += jnp.where((Sn < 0.0) | (Sn > 1e5), 1, 0)

        F = jnp.where((F < 0.0) | (F > 1e20), 0.0, F)
        error_count += jnp.where((F < 0.0) | (F > 1e20), 1000, 0)

        Jn = jnp.where((Jn < 0.0) | (Jn > 1e5), 0.0, Jn)
        error_count += jnp.where((Jn < 0.0) | (Jn > 1e5), 0.001, 0)

        T_Sn = jnp.abs(Sn)

        return Sn, Jn, F, T_Sn, error_count, key

    Sn, Jn, F, T_Sn, error_count, key = jax.lax.fori_loop(0, num_steps, loop_body, (Sn, Jn, F, T_Sn, error_count, key))
    
    return Sn, Jn, F, T_Sn, error_count, key
    
def rproc_loop(state, thetas, key, num_steps):

    Sn, Jn, F, T_Sn, error_count = state[0], state[1], state[2], state[3], state[4]
    sigSn, sigF, f_Sn, rn, k_Sn, sigJn, theta_Sn, theta_Jn, lambda_Jn = get_thetas(thetas)
    
    def loop_body_2(i, loop_state):
        Sn, Jn, F, T_Sn, error_count, main_key, sigSn, sigF, sigJn = loop_state
        dt = 0.25
        delta = 0.013
       
        main_key, key1, key2, key3 = jax.random.split(main_key, 4)
        
        noiSn = jax.random.normal(key = key1) * sigSn * jnp.sqrt(dt) 
        noiF = jax.random.normal(key = key2) * sigF * jnp.sqrt(dt) 
        noiJn = jax.random.normal(key = key3) * sigJn* jnp.sqrt(dt) 

        Sn_term = lambda_Jn * Jn * dt - theta_Sn * Sn * dt - delta * Sn * dt + Sn * noiSn
        Jn_term = rn * f_Sn * F * Sn * dt - lambda_Jn * Jn * dt - theta_Jn * Jn * dt - delta * Jn * dt + Jn * noiJn
        F_term = F * noiF - f_Sn * F * (Sn + 1 * Jn) * dt - delta * F * dt + 0.37 * dt

        Sn = Sn_term + Sn
        Jn = Jn_term + Jn
        F = F_term + F

        Sn = jnp.where((Sn < 0.0) | (Sn > 1e5), 0.0, Sn)
        error_count += jnp.where((Sn < 0.0) | (Sn > 1e5), 1, 0)

        F = jnp.where((F < 0.0) | (F > 1e20), 0.0, F)
        error_count += jnp.where((F < 0.0) | (F > 1e20), 1000, 0)

        Jn = jnp.where((Jn < 0.0) | (Jn > 1e5), 0.0, Jn)
        error_count += jnp.where((Jn < 0.0) | (Jn > 1e5), 0.001, 0)

        T_Sn = jnp.abs(Sn)
        return Sn, Jn, F, T_Sn, error_count, main_key, sigSn, sigF, sigJn

    Sn, Jn, F, T_Sn, error_count, key, sigSn, sigF, sigJn = jax.lax.fori_loop(0, num_steps, loop_body_2, (Sn, Jn, F, T_Sn, error_count, key, sigSn, sigF, sigJn))
    
    return Sn, Jn, F, T_Sn, error_count, key


def rproc_noi_rm(state, thetas, key, covars = None):
    # extract states and thetas
    day_index = state[5]
    int_day_index = day_index.astype(jnp.int32)
    dt = 0.25
    day_diff = covars[int_day_index]
    loop_num = day_diff/dt
    int_loop_num = loop_num.astype(jnp.int32)

    Sn, Jn, F, T_Sn, error_count, key = rproc_loop_noi_rm(state, thetas, key, int_loop_num)

    day_index = day_index + 1
    
    return jnp.array([Sn, Jn, F, T_Sn, error_count, day_index])


def rproc(state, thetas, key, covars = None):

    day_index = state[5]
    int_day_index = day_index.astype(jnp.int32)
    dt = 0.25
    day_diff = covars[int_day_index]
    loop_num = day_diff/dt
    int_loop_num = loop_num.astype(jnp.int32)

    Sn, Jn, F, T_Sn, error_count, key = rproc_loop(state, thetas, key, int_loop_num)

    day_index = day_index + 1
    
    return jnp.array([Sn, Jn, F, T_Sn, error_count, day_index])

rprocess_noi_rm = jax.vmap(rproc_noi_rm, (0, None, 0, None))
rprocesses_noi_rm = jax.vmap(rproc_noi_rm, (0 , 0, 0, None))

rprocess = jax.vmap(rproc, (0, None, 0, None))
rprocesses = jax.vmap(rproc, (0, None, 0, None))

def dnbinom_mu(y_val, k_Sn, T_Sn):
    p = k_Sn/(k_Sn + T_Sn)
    #logpmf(k, n, p, loc=0): n - number of success, p - probability of success
    return nbinom.logpmf(y_val, k_Sn, p)

def dmeas(y_val, state_preds, thetas):
    k_Sn = get_thetas(thetas)[4]
    T_Sn = state_preds[3]
    error_count = state_preds[4]
    log_lik = dnbinom_mu(y_val, k_Sn, T_Sn)
    log_lik_val = jnp.where(error_count>0, -150, log_lik)

    return log_lik_val

dmeasure = jax.vmap(dmeas, (None, 0, None))
dmeasures = jax.vmap(dmeas, (None, 0, 0))

def rinit(thetas, J, covars = None):
    Sn = 3
    F = 16.667
    Jn = 0
    T_Sn = 0.0
    error_count = 0.0

    day_index = 0

    return jnp.tile(jnp.array([Sn, Jn, F, T_Sn, error_count, day_index]), (J, 1))

