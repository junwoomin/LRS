"""Exercise real Pygame event handlers and simulator/model connections on the desktop."""
from argparse import Namespace
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import pygame
from gui import SimulatorUI

torch.set_num_threads(2)
out=Path('validation/gui_smoke')
args=Namespace(mode='simulate',town='Town03',seed=0,steps=64,rollout=4,output=out,autostart=False,snapshot=False)
ui=SimulatorUI(args)


def click(name):
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN,button=1,pos=ui.buttons[name].center))
    ui.frame()


def key(code):
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN,key=code))
    ui.frame()


try:
    assert not ui.error,ui.error
    assert all(m is not None for m in ui.fonts[17].metrics('시작새에피소드학습'))
    initial_route=np.asarray(ui.env.ego_path_px).copy()
    ui.screenshot('ready.png')
    click('run')
    for _ in range(3):ui.frame()
    assert ui.env.step_count==4
    key(pygame.K_SPACE)
    position=(ui.env.ego_vehicle.x,ui.env.ego_vehicle.y)
    steps=ui.env.step_count
    for _ in range(2):ui.frame()
    assert not ui.running and ui.env.step_count==steps
    assert position==(ui.env.ego_vehicle.x,ui.env.ego_vehicle.y)
    key(pygame.K_n)
    assert ui.env.step_count==steps+1
    ui.screenshot('paused.png')
    click('reset')
    assert ui.env.step_count==0 and not ui.running
    np.testing.assert_array_equal(initial_route,ui.env.ego_path_px)
    key(pygame.K_r)
    assert ui.env.step_count==0
    click('ppo')
    assert ui.mode=='ppo' and ui.agent is not None and not ui.running
    before={k:v.clone() for k,v in ui.agent.net.state_dict().items()}
    click('run')
    for _ in range(7):ui.frame()
    assert ui.updates==2 and ui.env.step_count==8,(ui.updates,ui.env.step_count,ui.error)
    key(pygame.K_SPACE)
    assert not ui.running
    changed=sum(not torch.equal(v,ui.agent.net.state_dict()[k]) for k,v in before.items())
    assert changed>0
    click('learn')
    assert ui.last_stats and np.isfinite(list(ui.last_stats.values())).all()
    ui.screenshot('learning.png')
    click('reward')
    ui.screenshot('rewards.png')
    ui.advance(action_override=[float('nan'),0.])
    assert ui.error and not ui.running
    ui.screenshot('error.png')
    click('reset')
    assert ui.error is None and ui.env.step_count==0
    pygame.event.post(pygame.event.Event(pygame.VIDEORESIZE,w=1024,h=760))
    ui.frame()
    assert (ui.width,ui.height)==(1024,760)
    click('drive')
    ui.screenshot('resized.png')
    click('screenshot')
    assert list(out.glob('screen-*.png'))
    ui.focus=None
    key(pygame.K_TAB);key(pygame.K_TAB);key(pygame.K_TAB);key(pygame.K_RETURN)
    # Tab traverses the mode selectors to Start; Enter begins simulation.
    assert ui.running or ui.env.step_count>=1
    key(pygame.K_ESCAPE)
    assert ui.quit_requested
finally:
    ui.close()
assert ui.closed and not pygame.display.get_init()
args.output=Path('validation/gui_restarted')
args.mode='simulate'
ui=SimulatorUI(args)
try:
    assert not ui.error
    click('run')
    assert ui.env.step_count==1
    ui.screenshot('restarted.png')
    click('quit')
    assert ui.quit_requested
finally:ui.close()
report={'status':'passed','pause_stops_physics':True,'single_step':True,'repeated_seeded_reset':True,
        'mode_switch':True,'ppo_updates':2,'changed_parameter_tensors':changed,'finite_learning_metrics':True,
        'saved_model':str(out/'ppo_latest.pth'),'error_display_and_recovery':True,
        'hangul_font_metrics':True,'resize':[1024,760],'keyboard_focus':True,'screenshot_button':True,
        'closed_and_reopened':True,'tracker_loaded':'wandb' in sys.modules}
assert 'wandb' not in sys.modules
(out/'ui_test_result.json').write_text(json.dumps(report,indent=2))
print('GUI_PASS',json.dumps(report))
