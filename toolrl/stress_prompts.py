"""Shared new tool contract and a fixed comprehensive demonstration candidate."""
import json

PROTOCOL='''Answer the current question using the private company database when necessary.
Emit exactly one JSON object per turn, without explanations or Markdown fences.
A lookup has exactly these keys: "action":"lookup", "entity":a known entity ID, "field":a field name.
An answer has exactly these keys: "action":"answer", "value":the exact requested scalar value.
Available fields by entity type:
project: budget, owner. The owner is an employee ID.
employee: department. The department is a department ID.
department: cost_center. The cost_center is a cost center ID.
cost_center: spending_limit. The spending_limit is a number.
Only a requested lookup reveals a private field. Copy entity IDs exactly from the current question or actual tool outputs. Never derive a private value from ID digits.
Work backwards from the requested field to decide which missing relationships are needed. Follow those relationships with one lookup at a time, then answer from the actual final result. An intermediate employee, department or cost center is not an answer when the question asks for a later field.
Facts explicitly supplied in the question are valid. Use them to start at the latest known point in the chain; do not re-query already supplied facts. If the final answer is supplied, answer immediately.
Tool errors consume a call and are not answers. If error is temporary_unavailable and retryable is true, retry the same entity and field. If error is entity_moved, retry the same field on the canonical_entity in that error. Never treat an error, owner ID or example value as the requested final value.
You have at most 6 lookup calls and 7 output turns, including the final answer. Minimize calls while answering correctly.
'''


def act(entity,field):
    return {'action':'lookup','entity':entity,'field':field}


def result(entity,field,value):
    return {'entity':entity,'field':field,'value':value}


def episode(question,pairs,answer):
    text='User: '+question+'\n'
    for action,observation in pairs:
        text+='Assistant: '+json.dumps(action,separators=(',',':'))+'\n'
        text+='Tool: '+json.dumps(observation,separators=(',',':'))+'\n'
    text+='Assistant: '+json.dumps({'action':'answer','value':answer},separators=(',',':'))+'\n'
    return text


def demonstrations():
    p,e,d,c='project_demo_a','employee_demo_a','department_74123','cost_center_demo_a'
    chain=[(act(p,'owner'),result(p,'owner',e)),(act(e,'department'),result(e,'department',d)),
           (act(d,'cost_center'),result(d,'cost_center',c))]
    text=episode('What is the cost center of the department of the owner of '+p+'?',chain,c)
    text+='\n'+episode('What spending limit applies to the cost center of the department of the owner of '+p+'?',
                       chain+[(act(c,'spending_limit'),result(c,'spending_limit',24761))],24761)
    text+='\n'+episode('What department does the owner of '+p+' belong to?',
        [chain[0],(act(e,'department'),{'error':'temporary_unavailable','entity':e,'field':'department','retryable':True}),chain[1]],d)
    alias='employee_demo_old'
    text+='\n'+episode('What department does the owner of project_demo_b belong to?',
        [(act('project_demo_b','owner'),result('project_demo_b','owner',alias)),
         (act(alias,'department'),{'error':'entity_moved','entity':alias,'field':'department','canonical_entity':e}),chain[1]],d)
    text+='\n'+episode(p+' is owned by '+e+'. What is the owner\'s department?',[chain[1]],d)
    text+='\n'+episode('The owner of '+p+' belongs to '+d+', whose cost center is '+c+'. What is the spending limit?',
                       [(act(c,'spending_limit'),result(c,'spending_limit',24761))],24761)
    text+='\n'+episode('The budget of '+p+' is 321 and its owner is '+e+'. Return its budget.',[],321)
    return text


PROMPTS={'protocol':PROTOCOL,'examples':PROTOCOL+'''
Complete illustrative episodes follow. Their entities and values are not facts for the current question.
Use the current question and actual tool results, never copy an example entity or value.
\n'''+demonstrations()+'''
The examples have ended. Solve the new current user question. Output only its next single JSON action.
'''}
