package tsml.classifiers.distance_based.optimised;

import evaluation.evaluators.Evaluator;
import evaluation.storage.ClassifierResults;
import tsml.classifiers.TSClassifier;
import tsml.classifiers.TrainTimeContractable;
import tsml.classifiers.distance_based.utils.classifiers.BaseClassifier;
import tsml.classifiers.distance_based.utils.collections.pruned.PrunedMultimap;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;

import java.util.*;

public class OptimisedClassifier extends BaseClassifier implements TrainTimeContractable {
    
    public OptimisedClassifier() {
        setExploiter(NO_EXPLOIT);
        setExplorer(NO_EXPLORE);
        setBatchSize(1000);
    }
    
    // method of obtaining new evaluations (exploring)
    private Explorer explorer;
    // method of improving previously seen evaluations (exploiting)
    private Exploiter exploiter;
    // the best evaluations (by default maximises score and retains the single best with tie breaks included)
    private final PrunedMultimap<Double, Evaluation> bestEvaluations = PrunedMultimap.descSoftSingle();
    // the limit for build time
    private long trainTimeLimit = -1;
    // the number of evaluations to evaluate in parallel
    private int batchSize;
    
    public int getBatchSize() {
        return batchSize;
    }

    public void setBatchSize(final int batchSize) {
        this.batchSize = batchSize;
        if(batchSize < 1) {
            throw new IllegalArgumentException("non pos batch size");
        }
    }


    // control the iteration through the optimisation space
    private boolean explore;
    private boolean exploit;
    private Set<Evaluation> allEvaluations;
    
    private boolean hasNext() {
        exploit = exploiter.hasNext();
        explore = explorer.hasNext();
        return exploit || explore;
    }

    private Evaluation next() {
        if(exploit && explore) {
            // could do either action
            // todo decide action
            
        }
        final Evaluation evaluation;
        if(exploit && !explore) {
            evaluation = exploiter.next();
        } else if(!exploit && explore) {
            evaluation = explorer.next();
        } else {
            throw new IllegalStateException("explore and exploit both false");
        }
        return evaluation;
    }
    
    @Override public double[] distributionForInstance(final TimeSeriesInstance inst) throws Exception {
        throw new UnsupportedOperationException();
    }

    @Override public void buildClassifier(final TimeSeriesInstances trainData) throws Exception {
        super.buildClassifier(trainData);
        boolean workDone = false;
        if(isRebuild()) {
            
        }
        while(hasNext()) {
            workDone = true;
            final List<Evaluation> batch = new LinkedList<>();
            do {
                batch.add(next());
            } while(hasNext());
            for(Evaluation evaluation : batch) {
                // todo make parallel
                final TSClassifier classifier = evaluation.getClassifier();
                final Evaluator evaluator = evaluation.getEvaluator();
                final ClassifierResults results = evaluator.evaluate(classifier, trainData);
                evaluation.setResults(results);
                exploiter.add(evaluation);
                // todo update best evaluations
            }
        }
        if(workDone) {
            
        }
//        // todo set the best evaluation - how to handle tie break?
//        // todo check best evaluation has been set, if not then random guess as too little time to do anything
//        // todo consider timing behaviour?
    }

    @Override public long getTrainContractTimeNanos() {
        return trainTimeLimit;
    }

    @Override public void setTrainTimeLimit(final long trainTimeLimit) {
        this.trainTimeLimit = trainTimeLimit;
    }

    public Explorer getExplorer() {
        return explorer;
    }

    public void setExplorer(final Explorer explorer) {
        this.explorer = Objects.requireNonNull(explorer);
    }

    public Exploiter getExploiter() {
        return exploiter;
    }

    public void setExploiter(final Exploiter exploiter) {
        this.exploiter = Objects.requireNonNull(exploiter);
    }


    public interface Exploiter extends Iterator<Evaluation> {
        void add(Evaluation evaluation);
    }
    
    public interface Explorer extends Iterator<Evaluation> {
        
    }
    
    private final static Exploiter NO_EXPLOIT = new Exploiter() {
        @Override public boolean hasNext() {
            return false;
        }

        @Override public Evaluation next() {
            throw new UnsupportedOperationException();
        }

        @Override public void add(final Evaluation evaluation) {
            // do nothing, we don't maintain the evaluations to be exploited because no exploitation is occurring
        }
    };
    private final static Explorer NO_EXPLORE = new Explorer() {
        @Override public boolean hasNext() {
            return false;
        }

        @Override public Evaluation next() {
            throw new UnsupportedOperationException();
        }
    };
}
