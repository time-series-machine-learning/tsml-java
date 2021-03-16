package tsml.classifiers.distance_based.optimised;

import evaluation.evaluators.Evaluator;
import evaluation.evaluators.InternalEstimateEvaluator;
import tsml.classifiers.TSClassifier;
import tsml.classifiers.distance_based.distances.dtw.spaces.DTWDistanceSpace;
import tsml.classifiers.distance_based.knn.KNN;
import tsml.classifiers.distance_based.utils.classifiers.configs.Builder;
import tsml.classifiers.distance_based.utils.classifiers.configs.ClassifierBuilder;
import tsml.classifiers.distance_based.utils.collections.checks.Checks;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;
import tsml.classifiers.distance_based.utils.collections.params.iteration.AbstractSearch;
import tsml.classifiers.distance_based.utils.collections.params.iteration.GridSearch;
import tsml.classifiers.distance_based.utils.collections.patience.Patience;
import tsml.data_containers.TimeSeriesInstances;

import java.util.*;

public class KnnAgent extends BaseAgent {

    public KnnAgent() {
        setNeighbourhoodSizeLimitProportional(1);
        setParamSpaceBuilder(new DTWDistanceSpace());
        setSearch(new GridSearch());
        setEvaluatorBuilder(InternalEstimateEvaluator::new);
        paramAgent.setClassifierBuilder((ClassifierBuilder<TSClassifier>) () -> {
            final KNN classifier = new KNN();
            classifier.setNeighbourhoodSizeLimit(neighbourhoodSize);
            classifier.setSeed(getSeed());
            return classifier;
        });
    }
    
    private double neighbourhoodSizeLimitProportional;
    private final BaseParamAgent paramAgent = new BaseParamAgent();
    private List<Evaluation> exploits;
    private List<Evaluation> nextExploits;
    private int maxNumParamSets;
    private int maxNeighbourhoodSize;
    private int neighbourhoodSize;
    private boolean usePatience;
    // the patience mechanisms for exploring / exploiting (may have different windows, etc, based on sensitivity)
    private final Patience explorePatience = new Patience(10);
    private final Patience exploitPatience = new Patience(100);
    private double bestExploitScore;
    private int exploitExpireCount;
    private int exploreExpireCount;
    private boolean exploreImprovement;
    private boolean exploitImprovement;
    private boolean explore;
    private final int expireCountThreshold = 2;
    
    public boolean isUsePatience() {
        return usePatience;
    }

    public void setUsePatience(final boolean usePatience) {
        this.usePatience = usePatience;
    }

    private boolean exploreOrExploitByPatience() {
        return explore;
    }

    private boolean exploreOrExploitByThreshold() {

        // need size of param space
        // need max neighbourhood size
        // then compare count of explores / exploits against limits

        // if less than 10% of param sets have been explored
        if(getEvaluations().size() < maxNumParamSets / 10) {
            // then explore more param sets
            return true;
        }
        // if less than 10% of neighbours have been looked at
        if(neighbourhoodSize < maxNeighbourhoodSize / 10) {
            // then exploit more neighbours
            return false;
        }

        // if less than 50% of param sets have been explored
        if(getEvaluations().size() < maxNumParamSets / 2) {
            // then explore more param sets
            return true;
        }
        // if less than 50% of neighbours have been looked at
        if(neighbourhoodSize < maxNeighbourhoodSize / 2) {
            // then exploit more neighbours
            return false;
        }

        // if less than 100% of param sets have been explored
        if(getEvaluations().size() < maxNumParamSets) {
            // then explore more param sets
            return true;
        }
        // if less than 100% of neighbours have been looked at
        if(neighbourhoodSize < maxNeighbourhoodSize) {
            // then exploit more neighbours
            return false;
        }

        throw new IllegalStateException("neighbours and param sets maxed out already");
    }
    
    @Override protected boolean exploreOrExploit() {
        if(usePatience) {
            return exploreOrExploitByPatience();
        } else {
            return exploreOrExploitByThreshold();
        }
    }

    @Override public void buildAgent(final TimeSeriesInstances trainData) {
        super.buildAgent(trainData);
        copyRandomTo(paramAgent);
        paramAgent.buildAgent(trainData);
        exploits = new LinkedList<>();
        nextExploits = new LinkedList<>();
        maxNumParamSets = paramAgent.getSearch().size();
        maxNeighbourhoodSize = trainData.numInstances();
        neighbourhoodSize = 1;
        if(usePatience) {
            exploitPatience.reset();
            explorePatience.reset();
            explorePatience.add(0);
            exploitPatience.add(0);
            bestExploitScore = 0;
            exploitImprovement = false;
            exploreImprovement = false;
            exploreExpireCount = 0;
            exploitExpireCount = 0;
            explore = true;
        }
    }

    @Override public void feedback(final Evaluation evaluation) {
        super.feedback(evaluation);
        if(!getClassifier(evaluation).isFullyBuilt()) {
            // can add more neighbours to improve the evaluation
            nextExploits.add(evaluation);
        }
        
        if(usePatience) {
            if(evaluation.isExplore() != explore) throw new IllegalStateException("received different type of evaluation than expected");
            if(evaluation.isExplore()) {
                exploreImprovement |= explorePatience.add(evaluation.getScore());
                if(explorePatience.isExpired()) {
                    exploitPatience.reset();
                    exploitPatience.add(explorePatience.getBest());
                    exploitImprovement = false;
                    bestExploitScore = 0;
                    if(exploreImprovement) {
                        exploreExpireCount = 0;
                        exploitExpireCount = 0;
                    } else {
                        exploreExpireCount++;
                    }
                    explore = false;
                }
            } else {
                bestExploitScore = Math.max(bestExploitScore, evaluation.getScore());
                if(exploits.isEmpty()) {
                    exploitImprovement |= exploitPatience.add(bestExploitScore);
                    if(exploitPatience.isExpired()) {
                        explorePatience.reset();
                        explorePatience.add(exploitPatience.getBest());
                        exploreImprovement = false;
                        if(exploitImprovement) {
                            exploitExpireCount = 0;
                            exploreExpireCount = 0;
                        } else {
                            exploitExpireCount++;
                        }
                        explore = true;
                    }
                }
            }
        }
    }
    
    @Override public boolean hasNextExploit() {
        return !(exploits.isEmpty() && nextExploits.isEmpty()) && exploitExpireCount < expireCountThreshold;
    }

    @Override protected Evaluation nextExploit() {
        if(exploits.isEmpty()) {
            exploits = nextExploits;
            nextExploits = new LinkedList<>();
        }
        final Evaluation evaluation = exploits.remove(0);
        evaluation.setResults(null); // clear the results (do not clear the score!)
        final KNN classifier = getClassifier(evaluation);
        int prevNeighbourhoodSize = classifier.getNeighbourhoodSize();
        // if the distance measure is symmetric then there's a 2 for the price of 1 on the neighbours
        final int increase = classifier.getDistanceMeasure().isSymmetric() ? 2 : 1;
        neighbourhoodSize = prevNeighbourhoodSize + increase;
        if(prevNeighbourhoodSize > neighbourhoodSize) {
            throw new IllegalStateException("neighbourhood size mismatch");
        }
        classifier.setNeighbourhoodSizeLimit(this.neighbourhoodSize);
        classifier.setRebuild(false); // stop the classifier from rebuilding from scratch (i.e. carry on from where it left off)
        return evaluation;
    }

    private KNN getClassifier(Evaluation evaluation) {
        final TSClassifier classifier = evaluation.getClassifier();
        if(!(classifier instanceof KNN)) {
            throw new IllegalStateException("expected knn");
        }
        return (KNN) classifier;
    }
    
    public boolean hasNextExploreParamSet() {
        return paramAgent.hasNextExplore();
    }
    
    @Override public boolean hasNextExplore() {
        return hasNextExploreParamSet() && exploreExpireCount < expireCountThreshold;
    }

    @Override protected Evaluation nextExplore() {
        return paramAgent.nextExplore();
    }

    public double getNeighbourhoodSizeLimitProportional() {
        return neighbourhoodSizeLimitProportional;
    }

    public void setNeighbourhoodSizeLimitProportional(final double neighbourhoodSizeLimitProportional) {
        this.neighbourhoodSizeLimitProportional = Checks.requireUnitInterval(neighbourhoodSizeLimitProportional);
    }
    
    public AbstractSearch getSearch() {
        return paramAgent.getSearch();
    }

    public ParamSpaceBuilder getParamSpaceBuilder() {
        return paramAgent.getParamSpaceBuilder();
    }

    public void setParamSpaceBuilder(
            final ParamSpaceBuilder paramSpaceBuilder) {
        paramAgent.setParamSpaceBuilder(paramSpaceBuilder);
    }

    public void setSearch(final AbstractSearch search) {
        paramAgent.setSearch(search);
    }

    public Builder<? extends Evaluator> getEvaluatorBuilder() {
        return paramAgent.getEvaluatorBuilder();
    }
    
    public void setEvaluatorBuilder(Builder<? extends Evaluator> builder) {
        paramAgent.setEvaluatorBuilder(builder);
    }

    public ResultsScorer getScorer() {
        return paramAgent.getScorer();
    }

    public void setScorer(final ResultsScorer scorer) {
        paramAgent.setScorer(scorer);
    }

    public Patience getExploitPatience() {
        return exploitPatience;
    }

    public Patience getExplorePatience() {
        return explorePatience;
    }
}
