import asyncio
import json
import sqlite3

import httpx
from society.client import LLMClient


def test_completion_budget_change_cannot_reuse_old_response(tmp_path):
    async def scenario():
        sent=[]
        def endpoint(request):
            sent.append(json.loads(request.content))
            return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'message':'Synthetic unit test','choice':'A','confidence':0.5,'cites':[]})}}]})
        model={'id':'unit-test','url':'http://unit-test/v1','fingerprint':'same-weights'}
        cache=tmp_path/'calls.sqlite'
        keys=[]
        for budget in [240,512]:
            client=LLMClient({**model,'max_completion_tokens':budget},cache)
            await client.http.aclose()
            client.http=httpx.AsyncClient(transport=httpx.MockTransport(endpoint))
            first=await client.complete('test','identical input',17)
            repeat=await client.complete('test','identical input',17)
            assert repeat['cached'] and repeat['call_key']==first['call_key']
            keys.append(first['call_key'])
            await client.close()
        assert len(set(keys))==2
        assert [x['max_tokens'] for x in sent]==[240,512]
        con=sqlite3.connect(cache)
        assert sorted(json.loads(r[0])['max_tokens'] for r in con.execute('select request from calls'))==[240,512]
        con.close()
    asyncio.run(scenario())


def test_explicit_grammar_hash_must_match_frozen_config(tmp_path):
    import hashlib
    import pytest
    from society.client import apply_output_constraints
    path=tmp_path/'test.ebnf';path.write_text('root ::= "test"')
    model={'structured_grammar_path':str(path),'structured_grammar_sha256':hashlib.sha256(path.read_text().encode()).hexdigest()}
    payload=apply_output_constraints({'response_format':{'type':'json_schema'}},model)
    assert payload=={'structured_outputs':{'grammar':path.read_text()}}
    path.write_text('root ::= "changed"')
    with pytest.raises(RuntimeError,match='hash mismatch'):
        apply_output_constraints({'response_format':{}},model)


def test_citation_length_constraint_keeps_invalid_ids_and_rejects_overflow(tmp_path,monkeypatch):
    async def scenario():
        async def no_delay(seconds):pass
        monkeypatch.setattr(asyncio,'sleep',no_delay)
        length=32
        requests=[]
        def endpoint(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'message':'Synthetic unit test','choice':'B','confidence':.5,'cites':['x'*length]})}}]})
        client=LLMClient({'id':'unit-test','url':'http://unit-test/v1','citation_max_length':32},tmp_path/'calls.sqlite')
        await client.http.aclose();client.http=httpx.AsyncClient(transport=httpx.MockTransport(endpoint))
        accepted=await client.complete('test','valid length but unknown ID',1)
        assert accepted['cites']==['x'*32]  # Does not repair/filter semantic citation errors.
        length=33
        import pytest
        with pytest.raises(RuntimeError,match='Citation exceeds'):
            await client.complete('test','too long',2)
        assert client.db.execute('SELECT COUNT(*) FROM failures').fetchone()[0]==4
        assert client.db.execute('SELECT COUNT(*) FROM calls').fetchone()[0]==1
        assert requests[0]['response_format']['json_schema']['schema']['properties']['cites']['items']['maxLength']==32
        from society.client import SCHEMA
        assert 'maxLength' not in SCHEMA['properties']['cites']['items']
        await client.close()
    asyncio.run(scenario())
