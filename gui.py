"""Pygame controls for fixed-policy simulation and bounded PPO training."""
import argparse
from dataclasses import asdict
import json
import math
import random
import time
import traceback
from pathlib import Path
import numpy as np
import torch
import pygame
from loop import BEVPathFollowEnv
from task_reward import TaskEnv, TaskRewardConfig
from roach.ppo import PPOAgent, PPOConfig, RolloutBuffer
from local_logging import LocalRun, run_directory

ROOT = Path(__file__).resolve().parent
PALETTE = {'background': '#eaf0f4', 'paper': '#ffffff', 'text': '#243d51',
           'muted': '#587183', 'blue': '#2678ab', 'soft_blue': '#d8eaf5',
           'line': '#cfdae2', 'road': '#8b9da9', 'map': '#dbe5eb',
           'ego': '#e36d48', 'danger': '#b54843', 'success': '#227866'}


class SimulatorUI:
    def __init__(self, args):
        self.args = args
        pygame.init()
        settings = json.loads((ROOT/'gui_settings.json').read_text())
        self.minimum_size = tuple(settings.get('minimum_window_size',[1024,760]))
        self.width, self.height = [max(size,minimum) for size,minimum in zip(settings['window_size'],self.minimum_size)]
        self.screen = pygame.display.set_mode((self.width,self.height),pygame.RESIZABLE)
        pygame.display.set_caption('LRS 주행 연구실')
        font = Path(settings['font_path'])
        if not font.exists():
            raise FileNotFoundError(f'한글 글꼴 경로를 gui_settings.json에서 확인하세요: {font}')
        self.fonts = {size:pygame.font.Font(str(font),size) for size in (13,15,17,19,24,30)}
        self.clock = pygame.time.Clock()
        self.output = args.output or run_directory(args.town)
        self.log = LocalRun(self.output,{**vars(args),'window':settings,'palette':PALETTE})
        self.mode = args.mode
        self.env = None
        self.agent = None
        self.buffer = None
        self.env_mode = None
        self.running = False
        self.ended = False
        self.quit_requested = False
        self.closed = False
        self.error = None
        self.message = '지도를 불러오는 중입니다'
        self.tab = 'drive'
        self.follow = True
        self.zoom = 1.6
        self.center = np.zeros(2)
        self.buttons = {}
        self.disabled = set()
        self.focus = None
        self.drag = None
        self.info = {}
        self.last_stats = None
        self.updates = 0
        self.total_steps = 0
        self.last_step_ms = 0.
        self.episode = 0
        self.draw()
        self.new_episode()
        self.running = args.autostart and not self.error

    def text(self, text, xy, size=17, color='text'):
        self.screen.blit(self.fonts[size].render(str(text),True,PALETTE.get(color,color)),xy)

    def event_log(self, event, **values):
        self.log.log({'ui/event':event,'mode':self.mode,**values},step=self.total_steps)

    def new_episode(self):
        self.running = False
        self.error = None
        try:
            seed = self.args.seed
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            if self.env is None or self.env_mode != self.mode:
                if self.env is not None:
                    self.env.close()
                cls = TaskEnv if self.mode=='ppo' else BEVPathFollowEnv
                kwargs = {'task_reward':TaskRewardConfig(max_episode_steps=self.args.steps)} if self.mode=='ppo' else {}
                self.env = cls(town=self.args.town,vis=False,render_mode='none',
                               device='cpu',FIXED_DT=1/30,sim_steps_per_frame=1,**kwargs)
                self.env_mode = self.mode
                self.map_surface = None
            # Constructors may consume random numbers; seed the actual episode reset.
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            self.obs,self.info = self.env.reset(seed=seed)
            if self.mode=='ppo':
                cfg = PPOConfig(device='cpu',rollout_steps=self.args.rollout,
                                update_epochs=1,minibatch_size=min(16,self.args.rollout),lr=1e-5)
                if self.agent is None:
                    self.agent = PPOAgent(self.obs['birdview'].shape,6,2,cfg)
                    self.agent.load(str(getattr(self.args,'checkpoint',Path('roach/log/ckpt_11833344.pth'))),strict=True)
                self.buffer = RolloutBuffer(self.args.rollout,self.obs['birdview'].shape,6,2,'cpu')
            self.ended = False
            self.follow = True
            self.drag = None
            self.last_step_ms = 0.
            self.episode += 1
            self.message = '준비됐습니다. 시작을 누르세요'
            self.event_log('reset',episode=self.episode,seed=seed)
        except Exception as exc:
            self.fail(exc)

    def fail(self, exc):
        self.running = False
        diagnostic = str(exc)
        self.error = ('가속과 조향은 -1~1 범위의 유한한 숫자여야 합니다'
                      if 'PPO action must contain' in diagnostic else diagnostic)
        self.message = '오류 내용을 확인한 뒤 새 에피소드로 다시 시작하세요'
        self.event_log('error',message=self.error,diagnostic=diagnostic)
        traceback.print_exc()

    def toggle_running(self):
        if self.error:
            return
        if self.ended:
            self.new_episode()
        self.running = not self.running
        self.message = '주행 중' if self.running else '일시정지 · 한 스텝씩 확인할 수 있습니다'
        self.event_log('start' if self.running else 'pause')

    def activate(self, name):
        if name=='run':self.toggle_running()
        elif name=='reset':self.new_episode()
        elif name=='step':
            if not self.running and not self.error and not self.ended:self.advance()
        elif name=='follow':self.follow=True
        elif name=='quit':self.quit_requested=True
        elif name=='screenshot':
            filename=f'screen-{time.time_ns()}.png'
            self.screenshot(filename)
            self.message=f'화면 저장: {filename}'
            self.event_log('screenshot',file=filename)
        elif name in ('simulate','ppo'):
            if name!=self.mode:
                self.mode=name
                self.new_episode()
        elif name in ('drive','reward','learn'):self.tab=name

    def handle_events(self):
        for event in pygame.event.get():
            if event.type==pygame.QUIT:self.quit_requested=True
            elif event.type==pygame.VIDEORESIZE:
                self.width,self.height=max(self.minimum_size[0],event.w),max(self.minimum_size[1],event.h)
                self.screen=pygame.display.set_mode((self.width,self.height),pygame.RESIZABLE)
            elif event.type==pygame.KEYDOWN:
                if event.key==pygame.K_ESCAPE:self.quit_requested=True
                elif event.key==pygame.K_SPACE:self.activate('run')
                elif event.key==pygame.K_r:self.activate('reset')
                elif event.key==pygame.K_n:self.activate('step')
                elif event.key==pygame.K_f:self.activate('follow')
                elif event.key==pygame.K_s:self.activate('screenshot')
                elif event.key==pygame.K_TAB:
                    names=[n for n in self.buttons if n not in self.disabled]
                    self.focus=names[(names.index(self.focus)+1)%len(names)] if self.focus in names else names[0]
                elif event.key==pygame.K_RETURN and self.focus:self.activate(self.focus)
                elif event.key in (pygame.K_LEFT,pygame.K_RIGHT,pygame.K_UP,pygame.K_DOWN):
                    self.follow=False
                    self.center+=np.array([{pygame.K_LEFT:-30,pygame.K_RIGHT:30}.get(event.key,0),
                                           {pygame.K_UP:-30,pygame.K_DOWN:30}.get(event.key,0)])/self.zoom
            elif event.type==pygame.MOUSEBUTTONDOWN and event.button==1:
                hit=next((name for name,rect in self.buttons.items() if name not in self.disabled and rect.collidepoint(event.pos)),None)
                if hit:self.focus=hit;self.activate(hit)
                elif self.map_rect.collidepoint(event.pos):self.drag=event.pos;self.follow=False
            elif event.type==pygame.MOUSEBUTTONUP and event.button==1:self.drag=None
            elif event.type==pygame.MOUSEMOTION and self.drag is not None:
                self.center-=np.asarray(event.rel)/self.zoom
                self.drag=event.pos
            elif event.type==pygame.MOUSEWHEEL:
                if self.map_rect.collidepoint(pygame.mouse.get_pos()):
                    self.zoom=float(np.clip(self.zoom*(1.12**event.y),.2,8.))

    def update_agent(self):
        if self.buffer.ptr<2:return
        self.message='정책을 갱신하는 중입니다'
        self.draw()
        ratios=[]
        with torch.no_grad():
            for bev,state,action,old_logp,*_ in self.buffer.get_batches(self.agent.cfg.minibatch_size,shuffle=False):
                dist,_=self.agent.net.get_dist_and_value(bev,state)
                ratios.extend(torch.exp(dist.log_prob(action)-old_logp).tolist())
        error=float(np.max(np.abs(np.asarray(ratios)-1)))
        if not math.isfinite(error) or error>1e-3:
            raise RuntimeError(f'갱신 전 정책 확률비 오류: {error}')
        self.buffer.compute_gae(0.,self.agent.cfg.gamma,self.agent.cfg.gae_lambda)
        self.last_stats=self.agent.update(self.buffer)
        self.updates+=1
        self.event_log('ppo_update',update=self.updates,preupdate_ratio_error=error,stats=self.last_stats)
        self.agent.save(str(self.output/'ppo_latest.pth'))
        self.buffer.reset()
        self.message='정책 갱신 완료 · 로컬 모델을 저장했습니다'

    def advance(self, action_override=None):
        if self.error or self.ended or self.env is None:return
        started=time.perf_counter()
        try:
            if self.mode=='ppo':
                action,logp,value=self.agent.act(self.obs)
                if action_override is not None:action=action_override
                next_obs,reward,terminated,truncated,info=self.env.step(action)
                nxt=0. if terminated else self.agent.value(next_obs)
                self.buffer.add(self.obs,action,logp,value,reward,terminated or truncated,
                                terminated=terminated,truncated=truncated,next_value=nxt)
            else:
                inp={k:torch.as_tensor(v).to(self.env.device).unsqueeze(0) for k,v in self.obs.items()}
                action=self.env._policy.forward(inp,deterministic=True,clip_action=True)[0]
                if action_override is not None:action=action_override
                next_obs,reward,terminated,info=self.env.step(action)
                truncated=self.env.step_count>=self.args.steps
                info.update({'collision':bool(self.env.veh_collision),'offroad':bool(self.env.das_collision)})
            self.obs,self.info=next_obs,info
            self.info['reward']=reward
            self.total_steps+=1
            self.log.log({'mode':self.mode,'episode':self.episode,'reward':reward,
                          'info':info,'action':action},step=self.total_steps)
            if self.mode=='ppo' and (self.buffer.ptr>=self.args.rollout or terminated or truncated):self.update_agent()
            if terminated or truncated:
                self.running=False;self.ended=True
                self.message=f'에피소드 종료: {info.get("reason") or "설정한 스텝 완료"}. 새로 시작할 수 있습니다'
        except Exception as exc:self.fail(exc)
        self.last_step_ms=(time.perf_counter()-started)*1000

    def draw_button(self, name, label, rect, selected=False, enabled=True):
        rect=pygame.Rect(rect);self.buttons[name]=rect
        if not enabled:self.disabled.add(name)
        fill=PALETTE['blue'] if selected else PALETTE['paper'] if enabled else '#dfe7ed'
        pygame.draw.rect(self.screen,fill,rect,border_radius=5)
        pygame.draw.rect(self.screen,PALETTE['line'],rect,1,border_radius=5)
        if self.focus==name:pygame.draw.rect(self.screen,PALETTE['ego'],rect.inflate(4,4),2,border_radius=5)
        rendered=self.fonts[15].render(label,True,PALETTE['paper'] if selected else PALETTE['text'])
        self.screen.blit(rendered,rendered.get_rect(center=rect.center))

    def draw_map(self):
        rect=self.map_rect
        pygame.draw.rect(self.screen,PALETTE['map'],rect)
        if self.env is None or self.env.ego_vehicle is None:return
        env=self.env
        if self.map_surface is None:
            canvas=np.empty_like(env.global_canvas)
            canvas[:]=pygame.Color(PALETTE['map'])[:3]
            road=np.any(env.global_canvas>0,axis=2)
            canvas[road]=pygame.Color(PALETTE['road'])[:3]
            lanes=np.all(env.global_canvas==255,axis=2)
            canvas[lanes]=(248,251,253)
            self.map_surface=pygame.surfarray.make_surface(canvas.swapaxes(0,1))
        ego=np.array([(env.ego_vehicle.x-env.OFFSET_X)*env.PPM,(env.ego_vehicle.y-env.OFFSET_Y)*env.PPM])
        if self.follow:self.center=ego
        view=np.array([rect.width,rect.height])/self.zoom
        top=np.maximum(0,np.minimum(self.center-view/2,np.maximum(0,np.array([env.MAP_W,env.MAP_H])-view)))
        source=pygame.Rect(int(top[0]),int(top[1]),max(1,min(int(view[0]),env.MAP_W-int(top[0]))),
                           max(1,min(int(view[1]),env.MAP_H-int(top[1]))))
        image=pygame.transform.smoothscale(self.map_surface.subsurface(source),rect.size)
        self.screen.blit(image,rect)
        scale=np.array(rect.size)/np.array(source.size)
        def screen_point(pixel):return tuple((np.array(rect.topleft)+(pixel-top)*scale).astype(int))
        self.screen.set_clip(rect)
        path=[screen_point(np.asarray(p)) for p in env.ego_path_px]
        if len(path)>1:pygame.draw.lines(self.screen,PALETTE['blue'],False,path,3)
        for light_id,patch in env.tl_patch_cache.items():
            state=env.tl_manager.state(light_id)
            color={'R':'#c85049','Y':'#daaa43','G':'#478b69'}.get(state,'#478b69')
            pygame.draw.line(self.screen,color,screen_point(np.array(patch['pt1'])),screen_point(np.array(patch['pt2'])),4)
        for vehicle,color in [(n['veh'],'#4e97bf') for n in env.npc_vehicle_agents if n.get('alive',True)]+[(env.ego_vehicle,PALETTE['ego'])]:
            px=np.array([(vehicle.x-env.OFFSET_X)*env.PPM,(vehicle.y-env.OFFSET_Y)*env.PPM])
            center=screen_point(px)
            body=pygame.Surface((max(5,int(4.69*env.PPM*scale[0])),max(3,int(2.*env.PPM*scale[1]))),pygame.SRCALPHA)
            body.fill(color)
            rotated=pygame.transform.rotate(body,-math.degrees(vehicle.yaw))
            self.screen.blit(rotated,rotated.get_rect(center=center))
        self.screen.set_clip(None)
        label='차량 따라가기' if self.follow else '지도 이동 중 · F로 복귀'
        label_rect=pygame.Rect(rect.x+12,rect.y+12,270,30)
        pygame.draw.rect(self.screen,PALETTE['paper'],label_rect,border_radius=4)
        self.text(f'{label}   {self.zoom:.1f}×',(label_rect.x+9,label_rect.y+6),15)

    def draw(self):
        self.screen.fill(PALETTE['background']);self.buttons={};self.disabled=set()
        self.text('LRS 주행 연구실',(20,16),24)
        status='오류' if self.error else ('학습 중' if self.mode=='ppo' else '주행 중') if self.running else '완료' if self.ended else '일시정지' if self.env and self.env.step_count else '준비'
        self.text(f'{self.args.town}  |  {status}',(260,24),17,'danger' if self.error else 'muted')
        self.draw_button('simulate','고정 정책 주행',(self.width-306,16,138,34),self.mode=='simulate')
        self.draw_button('ppo','PPO 학습',(self.width-156,16,136,34),self.mode=='ppo')
        label='일시정지' if self.running else '다시 시작' if self.ended else '시작'
        for name,text,x,w in [('run',label,20,116),('reset','새 에피소드',148,126),('step','한 스텝',286,96),
                              ('follow','차량 따라가기',394,130),('quit','종료',536,78),('screenshot','화면 저장',626,106)]:
            self.draw_button(name,text,(x,66,w,34),name=='run' and self.running,
                             enabled=not self.error if name=='run' else not (self.running or self.error or self.ended) if name=='step' else True)
        self.text('CPU · 로컬 기록',(self.width-158,76),15,'muted')
        self.map_rect=pygame.Rect(20,118,self.width-380,self.height-174)
        self.draw_map()
        x=self.width-338
        self.text('차량이 보는 BEV',(x,122),19)
        if self.env is not None and hasattr(self.env,'input_vis_rgb'):
            bev=pygame.surfarray.make_surface(self.env.input_vis_rgb.swapaxes(0,1))
            self.screen.blit(pygame.transform.scale(bev,(260,260)),(x,154))
        speed=float(self.info.get('speed',0.))
        self.text(f'{speed*3.6:4.1f} km/h',(x,427),30)
        reward_label = f'{self.info["reward"]:+.3f}' if 'reward' in self.info else '첫 스텝 대기'
        self.text(f'이번 스텝 보상  {reward_label}',(x,465),17)
        for name,label,offset in [('drive','주행',0),('reward','보상',102),('learn','학습',204)]:
            self.draw_button(name,label,(x+offset,500,94,30),self.tab==name)
        y=546
        rows=[]
        if self.tab=='drive' and self.env is not None:
            task=self.info.get('task',{})
            cte=task.get('cte_m',self.info.get('cte',0.))
            progress=task.get('route_completion',self.env.path_idx/max(1,len(self.env.ego_path)-1))*100
            rows=[('스텝 / 시간',f'{self.env.step_count} / {self.env.step_count*self.env.FIXED_DT:.2f}s'),
                  ('경로 진행',f'{progress:.2f}%'),('경로와 거리',f'{cte:.2f}m'),
                  ('충돌 / 도로 이탈',f'{"있음" if self.info.get("collision") else "없음"} / {"있음" if task.get("offroad",self.info.get("offroad")) else "없음"}'),
                  ('가속 / 제동',f'{self.env.throttle:.2f} / {self.env.brake:.2f}'),('조향',f'{self.env.steer:+.2f}'),
                  ('한 스텝 처리',f'{self.last_step_ms:.0f}ms')]
        elif self.tab=='reward':
            names={'progress':'새 경로 진행','time':'경과 시간','cte':'경로 이탈','heading':'방향 오차',
                   'overspeed':'과속','reverse':'후진','idle':'불필요한 정지','control_change':'제어 변화',
                   'collision':'충돌','offroute':'경로 벗어남','red_light':'빨간 신호 통과','stalled':'정체 종료','success':'경로 완료'}
            if self.mode=='simulate':rows=[('기존 시뮬레이션 보상',f'{self.info.get("reward",0.):+.3f}'),('PPO 보상 항목','학습 모드에서 확인')]
            else:rows=[(names.get(k,k),f'{v:+.4f}') for k,v in self.info.get('reward_terms',{}).items()]
            if not rows:rows=[('보상 항목','첫 스텝을 기다리는 중')]
        elif self.tab=='learn':
            if self.mode=='simulate':rows=[('고정 정책 주행','학습하지 않음'),('모드 변경','위의 PPO 학습 선택')]
            else:
                rows=[('정책 갱신',f'{self.updates}회'),('수집 스텝',f'{self.buffer.ptr if self.buffer else 0} / {self.args.rollout}')]
                names={'loss_pi':'정책 손실','loss_v':'가치 손실','entropy':'분포 엔트로피','approx_kl':'정책 변화 KL','lr':'학습률'}
                rows += [(names[k],f'{v:.5g}') for k,v in (self.last_stats or {}).items()]
                if self.last_stats is None:rows.append(('학습 지표','첫 갱신을 기다리는 중'))
        for label,value in rows:
            self.text(label,(x,y),15,'muted');self.text(value,(x+158,y),15);y+=24
        if self.error:
            box=pygame.Rect(self.map_rect.x+16,self.map_rect.bottom-100,self.map_rect.width-32,84)
            pygame.draw.rect(self.screen,'#fbe7e5',box,border_radius=5)
            self.text('실행이 멈췄습니다',(box.x+14,box.y+10),19,'danger')
            # Keep diagnostics within the map area; full details remain in local logs.
            limit=max(15,(box.width-28)//9)
            self.text(self.error[:limit],(box.x+14,box.y+40),15,'danger')
        self.text(self.message[:90],(20,self.height-46),15,'muted')
        self.text('Space 시작/멈춤   R 새 에피소드   N 한 스텝   F 따라가기   휠 확대   드래그 이동   S 화면 저장   Esc 종료',(20,self.height-24),13,'muted')
        pygame.display.flip()

    def frame(self):
        self.handle_events()
        if self.running and not self.quit_requested:self.advance()
        self.draw()
        self.clock.tick(30)

    def screenshot(self,name):
        self.draw();pygame.image.save(self.screen,str(self.output/name))

    def close(self):
        if self.closed:return
        if self.agent is not None and self.updates:self.agent.save(str(self.output/'ppo_latest.pth'))
        if getattr(self.args,'snapshot',False):self.screenshot('gui.png')
        if self.env is not None:self.env.close()
        self.event_log('close',steps=self.total_steps,updates=self.updates)
        (self.output/'summary.json').write_text(json.dumps({'steps':self.total_steps,'updates':self.updates,'closed':True,'error':self.error},indent=2))
        pygame.quit();self.closed=True

    def run(self):
        try:
            while not self.quit_requested:self.frame()
        finally:self.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description='LRS Pygame 주행/학습 제어판')
    parser.add_argument('--mode',choices=('simulate','ppo'),default='simulate')
    parser.add_argument('--town',default='Town03')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--steps',type=int,default=600,help='에피소드당 스텝 제한')
    parser.add_argument('--rollout',type=int,default=32)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--autostart',action='store_true')
    parser.add_argument('--snapshot',action='store_true')
    parser.add_argument('--checkpoint',type=Path,default=Path('roach/log/ckpt_11833344.pth'))
    args=parser.parse_args(argv)
    if args.steps<2 or args.rollout<2:parser.error('steps and rollout must be >=2')
    torch.set_num_threads(2)
    ui=SimulatorUI(args)
    ui.run()


if __name__=='__main__':main()
