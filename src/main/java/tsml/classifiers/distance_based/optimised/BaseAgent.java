package tsml.classifiers.distance_based.optimised;

import tsml.classifiers.distance_based.utils.system.logging.LogUtils;
import tsml.data_containers.TimeSeriesInstances;

import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;

public abstract class BaseAgent implements Agent {

    private transient Logger log = LogUtils.DEFAULT_LOG;
    // guard to make sure evaluations are selected / evaluated / fedback in proper order (i.e. no way to accidentally evaluate the same evaluation twice)
    private HashSet<Evaluation> pendingEvaluations;
    private boolean exploit = false;
    private boolean explore = false;
    private int evaluationIdCount;
    private int seed;
    private Random random = null;
    private List<Evaluation> evaluations;

    @Override public Logger getLogger() {
        return log;
    }

    @Override public void setLogger(final Logger logger) {
        this.log = Objects.requireNonNull(logger);
    }

    protected List<Evaluation> getEvaluations() {
        return evaluations;
    }

    @Override public List<Evaluation> getBestEvaluations() {
        final PrunedMap<Double, Evaluation> prunedMap = new PrunedMap<>();
        for(Evaluation evaluation : evaluations) {
            prunedMap.put(evaluation.getScore(), evaluation);
        }
        return prunedMap.valuesList();
    }

    @Override public void setSeed(final int seed) {
        this.seed = seed;
    }

    @Override public int getSeed() {
        return seed;
    }

    @Override public void setRandom(final Random random) {
        this.random = random;
    }

    @Override public Random getRandom() {
        return random;
    }

    protected Evaluation buildEvaluation() {
        return new Evaluation(evaluationIdCount++);
    }
    
    @Override public void buildAgent(final TimeSeriesInstances trainData) {
        checkRandom();
        evaluationIdCount = 0;
        pendingEvaluations = new HashSet<>();
        evaluations = new ArrayList<>();
    }

    protected boolean dropResultsOnFeedback() {
        return true;
    }
    
    @Override public void feedback(final Evaluation evaluation) {
        if(!pendingEvaluations.remove(evaluation)) {
            throw new IllegalStateException("unexpected feedback received for evaluation " + evaluation.getId());
        }
        
        // discard classifier results in the evaluation to decrease memory footprint - note this requires them to be 
        // recomputed if needed at a later date for train estimate, etc
        if(dropResultsOnFeedback()) {
            // depends on how expensive these are to recompute, sub classes can turn this off at the expense of memory
            evaluation.getResults().cleanPredictionInfo();
            evaluation.setResults(null);
        }
        
        // update the evaluations set
        // fill in any missing evaluations (although there shouldn't be any by the end, this is just if they get fed
        // back in the wrong order)
        for(int i = evaluations.size(); i <= evaluation.getId(); i++) {
            evaluations.add(null);
        }
        evaluations.set(evaluation.getId(), evaluation);
        
    }

    public abstract boolean hasNextExplore();
    
    public boolean hasNextExploit() {
        return false;
    }
    
    protected Evaluation nextExploit() {
        throw new UnsupportedOperationException();
    }
    
    protected abstract Evaluation nextExplore();
    
    @Override public boolean hasNext() {
        // either of two actions can be taken: exploit or explore. If neither, then we're done.
        exploit = hasNextExploit();
        explore = hasNextExplore();
        return exploit || explore;
    }

    @Override public Evaluation next() {
        if(!hasNext()) {
            throw new IllegalStateException("hasNext false");
        }
        // work out whether exploring / exploiting
        if(exploit && explore) {
            // must pick between exploring or exploit
            if(exploreOrExploit()) {
                exploit = false;
            } else {
                explore = false;
            }
        }
        final Evaluation evaluation;
        if(explore) {
            evaluation = nextExplore();
            evaluation.setExplore();
        } else if(exploit) {
            evaluation = nextExploit();
            evaluation.setExploit();
        } else {
            throw new IllegalStateException("both explore and exploit false");
        }
        if(!pendingEvaluations.add(evaluation)) {
            throw new IllegalStateException("already waiting on evaluation " + evaluation.getId());
        }
        return evaluation;
    }

    /**
     * 
     * @return true for explore, false for exploit
     */
    protected boolean exploreOrExploit() {
        return true; // by default always explore over exploit
    }

}
