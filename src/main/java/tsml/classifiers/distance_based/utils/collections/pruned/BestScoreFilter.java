package tsml.classifiers.distance_based.utils.collections.pruned;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class BestScoreFilter<B, A extends Comparable<A>> {

    public BestScoreFilter(final boolean higherIsBetter) {
        this.higherIsBetter = higherIsBetter;
    }
    
    public BestScoreFilter() {
        this(true);
    }

    private final boolean higherIsBetter;
    private List<B> best;
    private A bestScore;
    
    public boolean add(B obj, A score) {
        if(bestScore == null) {
            bestScore = score;
            best = new ArrayList<>(1);
            best.add(obj);
            return true;
        } else {
            int comparison = bestScore.compareTo(score);
            if(!higherIsBetter) {
                comparison = comparison * -1;
            }
            if(comparison <= 0) {
                if(comparison < 0) {
                    best = new ArrayList<>(1);
                    bestScore = score;
                }
                best.add(obj);
                return true;
            }
            return false;
        }
    }
    
    public List<B> getBest() {
        return Collections.unmodifiableList(best);
    }
    
}
