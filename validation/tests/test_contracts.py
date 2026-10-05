import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from roach.ppo import BetaActionDistribution, RolloutBuffer
from task_reward import TaskRewardConfig, reward_terms, credited_progress, crosses_segment


class RewardContracts(unittest.TestCase):
    def setUp(self):
        self.cfg=TaskRewardConfig()
        self.signals=dict(dt_s=1/30,new_progress_m=.15,cte_m=0.,heading_rad=0.,speed_m_s=5.,
                          collision=False,offroad=False,red_light=False,idle_seconds=0.,
                          idle=False,allowed_stop=False,success=False,action_change_squared=0.)

    def reward(self,**updates):
        return reward_terms({**self.signals,**updates},self.cfg)

    def test_stationary_center_has_no_positive_reward(self):
        terms,term,_=self.reward(new_progress_m=0.,speed_m_s=0.,idle=True)
        self.assertLess(sum(terms.values()),0.)
        self.assertFalse(term)

    def test_speed_alone_has_no_reward(self):
        terms,_,_=self.reward(new_progress_m=0.,speed_m_s=30.)
        self.assertLess(sum(terms.values()),0.)

    def test_collision_suppresses_progress_success_and_duplicate_cost(self):
        terms,term,reason=self.reward(collision=True,offroad=True,red_light=True,success=True,new_progress_m=10.)
        self.assertEqual(terms,{'collision':-self.cfg.collision_cost})
        self.assertTrue(term);self.assertEqual(reason,'collision')

    def test_road_and_route_failure_exclusive(self):
        for fields in ({'offroad':True,'cte_m':5.},{'cte_m':5.}):
            terms,term,reason=self.reward(**fields)
            self.assertEqual(terms,{'offroute':-self.cfg.offroute_cost})
            self.assertTrue(term)

    def test_idle_timeout_and_allowed_red_wait(self):
        terms,term,reason=self.reward(new_progress_m=0.,speed_m_s=0.,idle=True,idle_seconds=5.)
        self.assertEqual(reason,'stalled');self.assertTrue(term)
        terms,term,reason=self.reward(new_progress_m=0.,speed_m_s=0.,idle=True,allowed_stop=True)
        self.assertEqual(terms['idle'],0.)
        self.assertLessEqual(sum(terms.values()),0.)
        self.assertFalse(term)

    def test_time_units_and_control_cost(self):
        a,_,_=self.reward(new_progress_m=0.,cte_m=1.,dt_s=.1)
        b,_,_=self.reward(new_progress_m=0.,cte_m=1.,dt_s=.2)
        self.assertAlmostEqual(sum(b.values()),2*sum(a.values()))
        smooth,_,_=self.reward(action_change_squared=2.)
        steady,_,_=self.reward()
        self.assertLess(sum(smooth.values()),sum(steady.values()))

    def test_progress_oscillation_and_teleport(self):
        frontier=0.
        credited=[]
        for s in (1.,0.,1.,0.,1.):
            value,frontier=credited_progress(s,frontier,1.,1.,self.cfg)
            credited.append(value)
        self.assertEqual(credited,[1.,0.,0.,0.,0.])
        value,_=credited_progress(100.,0.,.1,1/30,self.cfg)
        self.assertLessEqual(value,.125)

    def test_red_crossing_and_stationary_line(self):
        a,b=np.array([0.,-1.]),np.array([0.,1.])
        c,d=np.array([-1.,0.]),np.array([1.,0.])
        self.assertTrue(crosses_segment(a,b,c,d))
        self.assertFalse(crosses_segment(c,c,c,d))

    def test_progress_and_success(self):
        terms,term,_=self.reward()
        self.assertGreater(sum(terms.values()),0.)
        terms,term,reason=self.reward(success=True)
        self.assertTrue(term);self.assertEqual(reason,'success')


class PPOContracts(unittest.TestCase):
    def test_bounded_beta_logprob_and_entropy_transform(self):
        alpha=torch.tensor([[2.,3.]])
        beta=torch.tensor([[3.,2.]])
        dist=BetaActionDistribution(alpha,beta)
        action=torch.tensor([[-.2,.4]])
        expected=dist.base_dist.log_prob((action+1)/2).sum(-1)-2*math.log(2)
        torch.testing.assert_close(dist.log_prob(action),expected)
        expected_entropy=dist.base_dist.entropy().sum(-1)+2*math.log(2)
        torch.testing.assert_close(dist.entropy(),expected_entropy)
        samples=torch.stack([dist.sample() for _ in range(100)])
        self.assertTrue(torch.all((samples>-1)&(samples<1)))
        self.assertTrue(torch.isfinite(dist.log_prob(samples)).all())

    def test_terminated_vs_truncated_bootstrap(self):
        obs={'birdview':np.zeros((1,1,1),np.float32),'state':np.zeros(1,np.float32)}
        for terminal,expected in ((True,1.),(False,9.1)):
            buf=RolloutBuffer(1,(1,1,1),1,2,'cpu')
            buf.add(obs,[0,0],0,2,1,True,terminated=terminal,truncated=not terminal,next_value=9.)
            buf.compute_gae(100.,.9,.95)
            self.assertAlmostEqual(float(buf.ret[0]),expected,places=5)

    def test_trace_does_not_leak_across_reset(self):
        obs={'birdview':np.zeros((1,1,1),np.float32),'state':np.zeros(1,np.float32)}
        buf=RolloutBuffer(2,(1,1,1),1,2,'cpu')
        buf.add(obs,[0,0],0,2,1,True,terminated=False,truncated=True,next_value=9.)
        buf.add(obs,[0,0],0,100,20,True,terminated=True,next_value=0.)
        buf.compute_gae(0,.9,.95)
        self.assertAlmostEqual(float(buf.ret[0]),9.1,places=5)
        self.assertAlmostEqual(float(buf.ret[1]),20.,places=5)


if __name__=='__main__':
    unittest.main(verbosity=2)
