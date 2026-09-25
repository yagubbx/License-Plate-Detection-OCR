from .geometry import iou

def edit_distance(a,b):
    previous=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        current=[i]
        for j,cb in enumerate(b,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(ca != cb)))
        previous=current
    return previous[-1]

def match_boxes(predictions, truth, threshold=0.5):
    """Confidence-ordered, one-to-one matching. A GT can be claimed once."""
    unused=set(range(len(truth)))
    matches=[]
    for pi in sorted(range(len(predictions)),key=lambda i:predictions[i]['confidence'],reverse=True):
        if not unused:
            break
        gi=max(sorted(unused),key=lambda j:iou(predictions[pi]['box'],truth[j]['box']))
        overlap=iou(predictions[pi]['box'],truth[gi]['box'])
        if overlap >= threshold:
            matches.append((pi,gi,overlap)); unused.remove(gi)
    return matches
