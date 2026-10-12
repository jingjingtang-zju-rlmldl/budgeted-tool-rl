"""One frozen contract; comprehensive demonstration control is fixed in advance."""
import json

PROTOCOL = '''Find the effective spending limit of the specified project using actual tool results or explicitly supplied facts.
Emit exactly one JSON object per turn. No prose or Markdown.
Lookup: {"action":"lookup","entity":"known ID","field":"field name"}.
Answer: {"action":"answer","value":exact requested number}.
Fields: project: policy, owner, partner, local_limit, resolved_limit, budget; employee: department; department: cost_center; cost_center: spending_limit; vendor: spending_limit.
The project policy determines its effective limit. local: project.local_limit. team: project.owner -> employee.department -> department.cost_center -> cost_center.spending_limit. partner: project.partner -> vendor.spending_limit. Read policy before choosing a branch unless policy is supplied.
Alternative: project.resolved_limit directly returns the correct effective limit for any policy, without needing policy or the owner chain. Its query costs the service price explicitly quoted in the current question (2 or 6). Every other query costs 1. Errors and repeated queries also cost money. Tool outputs disclose query_cost and total_cost.
Minimize total query cost while answering correctly. If costs tie, prefer fewer calls. Use supplied facts and previous actual results to skip known relationships. A supplied final effective limit needs no query. Do not return an unused branch's value or the project budget.
Copy the complete entity ID exactly from the current question or actual observations; retain every character. If unknown_entity occurs, recheck the original known ID and correct your copy. There is no automatic ID hint. If unknown_field occurs, consult allowed_fields and the schema; an error is never an answer. Never invent fields from question words.
At most 6 lookup calls and 7 output turns including the answer. Output only the next single action.
'''


def demos():
    def episode(q, calls, value):
        out='User: '+q+'\n'; total=0
        for e,f,v,c in calls:
            total += c
            out += 'Assistant: '+json.dumps(dict(action='lookup',entity=e,field=f),separators=(',',':'))+'\n'
            obs = (dict(error='unknown_entity',entity=e,field=f) if v is None else dict(entity=e,field=f,value=v))
            obs.update(query_cost=c,total_cost=total)
            out += 'Tool: '+json.dumps(obs,sort_keys=True)+'\n'
        return out+'Assistant: '+json.dumps(dict(action='answer',value=value),separators=(',',':'))+'\n'
    p,e,d,c,v='project_demo_t','employee_demo_t','department_demo_t','cost_center_demo_t','vendor_demo_t'
    chain=[(p,'policy','team',1),(p,'owner',e,1),(e,'department',d,1),(d,'cost_center',c,1),(c,'spending_limit',47261,1)]
    text=episode('Effective limit of '+p+'? Service price 6.',chain,47261)
    text+=episode('Effective limit of '+p+'? Service price 2.',[(p,'resolved_limit',47261,2)],47261)
    text+=episode('Effective limit of project_demo_p? Service price 6.',[('project_demo_p','policy','partner',1),('project_demo_p','partner',v,1),(v,'spending_limit',68312,1)],68312)
    text+=episode('Effective limit of project_demo_l? Service price 6.',[('project_demo_l','policy','local',1),('project_demo_l','local_limit',29418,1)],29418)
    text+=episode('Effective limit of '+p+'? Policy team, owner '+e+', owner department '+d+'. Service price 6.',[(d,'cost_center',c,1),(c,'spending_limit',47261,1)],47261)
    text+=episode('Effective limit of '+p+'? Policy local. Service price 2.',[(p,'local_limit',29418,1)],29418)
    text+=episode('Effective limit of '+p+'? Supplied effective limit 47261. Service price 6.',[],47261)
    text+=episode('Effective limit of project_demo_copy? Service price 2.',[('project_demo_cop','resolved_limit',None,2),('project_demo_copy','resolved_limit',83725,2)],83725)
    return text


PROMPTS = {'protocol':PROTOCOL,'examples':PROTOCOL+'\nIllustrative episodes; their values and IDs are not current facts.\n'+demos()+'\nExamples end. Solve the current question with only its next single action.\n'}
