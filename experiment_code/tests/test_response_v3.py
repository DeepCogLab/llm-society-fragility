import asyncio
import copy
import json
import re

import pytest

from society.core import Spec, digest
from society.response_v3 import (load_config, profile_cells, profile_context, private_prompt,
    select_condition, fit_predictor, predictor_features, simulate_network, run_network_arm, write_once)


def cell(**changes):
    return {"stage":"calibration","replicate":60000,"q":.7,"private_correct":True,
            "support":3,"kind":"report","interface":"provenance_shared","history":"aligned",**changes}


def test_fixed_counts_disjoint_tasks_and_unique_conditions():
    config=load_config()
    a=list(profile_cells(config,"calibration"));b=list(profile_cells(config,"validation"))
    assert len(a)==2952 and len(b)==5904
    assert len({digest(x) for x in a})==len(a)
    assert not {x['replicate'] for x in a}&{x['replicate'] for x in b}
    assert len([x for x in a if x['private_correct']])==len(a)//2


def test_both_private_states_and_explicit_source_dependence():
    for correct in [True,False]:
        _,task,private,inbox,prior,prompt=profile_context(cell(private_correct=correct))
        assert (private==task['truth'])==correct
        assert sum(m['choice']!=private for m in inbox)==3
        assert len({m['roots'][0] for m in inbox})==2
        assert len(prior)==1 and prior[0]['choice']==private
        assert 'Declared roots' in prompt
        assert 'private_correct' not in prompt and 'ground truth' not in prompt
    _,_,_,inbox,_,_=profile_context(cell(interface='provenance_independent'))
    assert len({m['roots'][0] for m in inbox})==4


def test_history_manipulation_and_private_visibility_do_not_touch_neighbors():
    spec,task,private,inbox,prior,_=profile_context(cell())
    p0=private_prompt(spec,task,0,0,inbox,prior,'initial_only')
    p1=private_prompt(spec,task,0,1,inbox,prior,'initial_only')
    pp=private_prompt(spec,task,0,1,inbox,prior,'persistent')
    assert f'observes {private}; reliability' in p0
    assert f'observes {private}; reliability' not in p1
    assert f'observes {private}; reliability' in pp
    assert p1[p1.index('Messages delivered this round:'):]==pp[pp.index('Messages delivered this round:'):]
    assert profile_context(cell(history='opposed'))[4][0]['choice']!=private


def test_selection_uses_only_calibration_and_does_not_force_a_condition():
    config=load_config();records=[]
    for q in config['reliabilities']:
        for correct in [True,False]:
            for i in range(4):
                changed=(i<2) if q==.55 else False
                records.append({**cell(q=q,private_correct=correct,interface='natural'),"wrong":changed if correct else not changed,"changed_private":changed})
            records.append({**cell(q=q,private_correct=correct,support=-1),"wrong":not correct,"changed_private":False})
    result=select_condition(records,config)
    assert result['selected_q']==.55
    assert result['network_qs']==[.55,.7]
    contaminated=copy.deepcopy(records);contaminated[0]['stage']='validation'
    with pytest.raises(ValueError):select_condition(contaminated,config)
    for row in records:
        if row['support']!=-1:
            row['wrong']=not row['private_correct'];row['changed_private']=False
    result=select_condition(records,config)
    assert result['selected_q'] is None and result['network_qs']==[.7]


def test_predictor_rejects_heldout_training_and_features_do_not_use_truth():
    with pytest.raises(ValueError):fit_predictor([{'stage':'validation'}],[])
    private='A';inbox=[{'choice':'B','injected':True}];prior=[{'choice':'A'}]
    a=predictor_features(private,.7,inbox,prior,True)
    assert len(a)==11 and a[0]==1 and a[2]==1 and a[6]==0
    assert predictor_features(private,.7,inbox,prior,False)[7]==0


def test_no_update_forward_baseline_has_no_invented_contagion():
    spec=Spec(phase='test-v3',n=8,degree=4,rounds=5,onset=2,duration=1)
    results=simulate_network(spec,'persistent',{'constant':0.0},2)
    assert len(results)==6
    assert all(r['curve']==results[0]['curve'] for r in results)


class FakeClient:
    model={'id':'test','family':'test','fingerprint':'test-v3'}
    async def complete(self,system,prompt,seed,temperature):
        answers=re.findall(r'answer=([AB]);',prompt)
        private=re.search(r'Your private sensor report E\d+ observes ([AB]);',prompt)
        if answers:
            choice='A' if answers.count('A')>=answers.count('B') else 'B'
        else:
            choice=private[1] if private else 'A'
        own=re.search(r'You are peer (\d+)',prompt)[1]
        return {'choice':choice,'confidence':.7,'message':f'My answer is {choice}.','cites':[f'E{int(own):03d}'],'call_key':digest([prompt,seed])}


def test_network_pairing_window_and_no_relay_indirect_invariance(tmp_path):
    spec=Spec(phase='test-v3-network',replicate=9,n=8,degree=4,rounds=5,onset=2,duration=1)
    async def run():
        client=FakeClient()
        for schedule in ['persistent','initial_only']:
            base=await run_network_arm(client,spec,schedule,'control',None,tmp_path/schedule/'control.json')
            tr=await run_network_arm(client,spec,schedule,'treated',base,tmp_path/schedule/'treated.json')
            nr=await run_network_arm(client,spec,schedule,'no_relay_treated',base,tmp_path/schedule/'no_relay.json')
            for c,e in zip(base['events'],tr['events']):
                if e['round']<2:assert c['call_key']==e['call_key']
                if e['direct_injected_exposure']:assert e['round']==2
            for c,e in zip(base['events'],nr['events']):
                if e['distance_from_seeds'] is None or e['distance_from_seeds']>=2:
                    assert c['call_key']==e['call_key']
    asyncio.run(run())


def test_immutable_artifacts_cannot_be_replaced(tmp_path):
    path=tmp_path/'frozen.json';write_once(path,{'q':.7});write_once(path,{'q':.7})
    with pytest.raises(RuntimeError):write_once(path,{'q':.55})
    assert json.loads(path.read_text())=={'q':.7}


def test_complete_small_pipeline_reports_and_frozen_prediction(tmp_path,monkeypatch):
    import society.response_v3 as v3
    monkeypatch.setattr(v3,'OUTPUT',tmp_path)
    config=copy.deepcopy(load_config())
    for stage in ['calibration','validation','sequence_calibration','sequence_validation']:
        config[stage]['reps']=2
    config['network'].update({'n':8,'reps':2,'topologies':['WS'],'simulation_mc':1})
    async def run():
        client=FakeClient();model=client.model['id']
        await v3.run_profile(client,config,'calibration')
        records=v3.read_profile(model,'calibration',492)
        selection=v3.select_condition(records,config)
        await v3.run_sequence(client,config,'sequence_calibration',selection['network_qs'])
        sequences=v3.read_sequences(model,'sequence_calibration',16*len(selection['network_qs']))
        predictor=v3.fit_predictor(records,sequences)
        predictions=v3.freeze_predictions(model,config,selection,predictor)
        before=(tmp_path/'models'/model/'predictions_before_validation.json').read_bytes()
        await v3.run_profile(client,config,'validation')
        await v3.run_sequence(client,config,'sequence_validation',selection['network_qs'])
        v3.profile_report(model,config,selection)
        await v3.run_network(client,config,selection)
        v3.network_report(model,config,selection,predictions)
        assert (tmp_path/'models'/model/'network_tail_effects.csv').exists()
        assert (tmp_path/'models'/model/'forward_prediction_evaluation.csv').exists()
        assert before==(tmp_path/'models'/model/'predictions_before_validation.json').read_bytes()
        integrity=json.loads((tmp_path/'models'/model/'network_integrity.json').read_bytes())
        assert integrity['preperiod_mismatches']==0 and integrity['outside_window']==0
    asyncio.run(run())


def test_freeze_creates_relative_snapshot_without_touching_sources(tmp_path,monkeypatch):
    import society.response_v3 as v3
    from pathlib import Path
    original=Path('society/response_v3.py').read_bytes()
    monkeypatch.setattr(v3,'OUTPUT',tmp_path)
    record=v3.freeze(load_config(),{'test':{'id':'test'}})
    assert all(not Path(p).is_absolute() for p in record['sources'])
    assert (tmp_path/'source_snapshot/society/response_v3.py').read_bytes()==original
    assert Path('society/response_v3.py').read_bytes()==original
    assert v3.check_frozen()['hash']==record['hash']
