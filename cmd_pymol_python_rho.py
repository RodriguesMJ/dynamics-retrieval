from pymol import cmd
      
def myfunc_step():    
    cmd.load('../bov_nlsa_refine_39_chainB.pdb')
    cmd.color('blue', selection=' (name C*)')
    #cmd.load('./1ps_chainA_superposed.pdb')
    cmd.show('sticks', 'all')
    cmd.bg_color('white')
    cmd.hide('all')
    #cmd.select('sel', '((chain A and (resi 212 or resi 216 or resn RET) and not h.) or (chain A and (resi 407 or resi 411 or resi 415)))')
    cmd.select('sel', 'resn RET')
    cmd.show('lines', 'sel')    
    cmd.show('spheres', 'sel')    
    cmd.set('sphere_scale', 0.10, 'all')
    
    for time in range(0, 138300, 100):
        cmd.load('./2.0_rho_light_mode_0_1_timestep_%0.6d_light--dark_rho_dark_mode_0_avg.ccp4'%(time), 'nlsa_map')
        cmd.zoom('sel')
        # RETA
        #cmd.set_view ('\
        #               0.080371954,    0.806613684,   -0.585589767,\
        #              -0.930780470,   -0.149466500,   -0.333627909,\
        #              -0.356634021,    0.571868539,    0.738767087,\
        #               0.000000000,    0.000000000,  -38.521888733,\
        #               9.948150635,   28.112447739,   29.300647736,\
        #              30.370950699,   46.672828674,  -20.000000000 ')

        # RET B
        cmd.set_view ('\
                        0.203187689,    0.868704140,   -0.451742560,\
                        0.946889222,   -0.056905121,    0.316471487,\
                        0.249215886,   -0.492051393,   -0.834131420,\
                        0.000000000,    0.000000000,  -39.938152313,\
                      -20.810651779,   28.454404831,   46.120998383,\
                       31.775022507,   48.101264954,  -20.000000000 ')
        cmd.isomesh('map_modes_0_x_p', 'nlsa_map', 3.0, 'sel', carve=2.0)
        cmd.color('cyan', 'map_modes_0_x_p')
        cmd.isomesh('map_modes_0_x_m', 'nlsa_map', -3.0, 'sel', carve=2.0)
        cmd.color('purple', 'map_modes_0_x_m')
        cmd.set('mesh_width', 0.3)
        #cmd.set('fog_start', 0.1)
        cmd.show('mesh', 'map_modes_0_x_p')
        cmd.show('mesh', 'map_modes_0_x_m')
        cmd.ray(2048, 1024)
        cmd.png('./FRAMES_light_mode_0_1_minus_dark_avg_mode_0_3p0s_RETB/nlsa_modes_0_1_time_%d.png'%(time))
        
        cmd.delete('map_modes_0_x_p')
        cmd.delete('map_modes_0_x_m')
        cmd.delete('nlsa_map')

def myfunc_step_pocket():    
    cmd.load('../bov_nlsa_refine_39_chainB.pdb')
    cmd.color('blue', selection=' (name C*)')
    #cmd.load('./1ps_chainA_superposed.pdb')
    cmd.show('sticks', 'all')
    cmd.bg_color('white')
    cmd.hide('all')
    #cmd.select('sel', '((chain A and (resi 212 or resi 216 or resn RET) and not h.) or (chain A and (resi 407 or resi 411 or resi 415)))')
    cmd.select('sel', 'resi 113 or resi 118 or resi 186 or resi 207 or resi 212 or resi 265 or resi 268 or resi 296 or resn RET')
    cmd.show('lines', 'sel')    
    cmd.show('spheres', 'sel')    
    cmd.set('sphere_scale', 0.10, 'all')
    
    for time in range(118500, 118600, 100): #138300, 100):
        cmd.load('./2.0_rho_light_mode_0_2_timestep_%0.6d_light--dark_rho_dark_mode_0_avg.ccp4'%(time), 'nlsa_map')
        cmd.zoom('sel')
        # RETA
        #cmd.set_view ('\
        #               0.080371954,    0.806613684,   -0.585589767,\
        #              -0.930780470,   -0.149466500,   -0.333627909,\
        #              -0.356634021,    0.571868539,    0.738767087,\
        #               0.000000000,    0.000000000,  -38.521888733,\
        #               9.948150635,   28.112447739,   29.300647736,\
        #              30.370950699,   46.672828674,  -20.000000000 ')

        # RET B
        cmd.set_view ('\
                        0.203187689,    0.868704140,   -0.451742560,\
                        0.946889222,   -0.056905121,    0.316471487,\
                        0.249215886,   -0.492051393,   -0.834131420,\
                        0.000000000,    0.000000000,  -39.938152313,\
                      -20.810651779,   28.454404831,   46.120998383,\
                       31.775022507,   48.101264954,  -20.000000000 ')
        cmd.isomesh('map_modes_0_x_p', 'nlsa_map', 3.0, 'sel', carve=2.0)
        cmd.color('cyan', 'map_modes_0_x_p')
        cmd.isomesh('map_modes_0_x_m', 'nlsa_map', -3.0, 'sel', carve=2.0)
        cmd.color('purple', 'map_modes_0_x_m')
        cmd.set('mesh_width', 0.3)
        #cmd.set('fog_start', 0.1)
        cmd.show('mesh', 'map_modes_0_x_p')
        cmd.show('mesh', 'map_modes_0_x_m')
        cmd.ray(2048, 1024)
        cmd.png('./FRAMES_light_mode_0_2_minus_dark_avg_mode_0_3p0s_RETB/nlsa_modes_0_1_time_%d.png'%(time))
        
        cmd.delete('map_modes_0_x_p')
        cmd.delete('map_modes_0_x_m')
        cmd.delete('nlsa_map')
        
cmd.extend('myfunc_step',  myfunc_step)
cmd.extend('myfunc_step_pocket',  myfunc_step)

print "\n**** CALLING myfunc_step ****"
myfunc_step_pocket()  
