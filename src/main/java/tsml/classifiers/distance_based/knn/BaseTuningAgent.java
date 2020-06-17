package tsml.classifiers.distance_based.knn;

import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Purpose: explore a classifier space. This could be anything, but simple use-case would be exploring a parameter
 * set. Suppose I have a KNN with DTW and a window parameter ranging from 0..99. KNNs can be restricted to carry out
 * LOOCV on only a subset of the train data. It would be very expensive to get every DTW parameter with a full LOOCV
 * conducted, therefore we could cut corners by skipping DTW parameters OR evaluating the effectiveness of a
 * parameter using a restricted LOOCV (i.e. less neighbours). How exactly all this is done is up to you, but this
 * class assumes there are two cases:
 *      - explore: find an unseen DTW parameter and build a KNN, evaluate using a minimal LOOCV
 *      - exploit: find a previously seen DTW KNN and raise the number of neighbours, increasing the quality of the
 *      LOOCV
 *
 * In essense, you should be able to explore a "classifier space" in a restricted manner.
 */
public abstract class BaseTuningAgent implements TuningAgent {

    /**
     * generate a new classifier with a new param set, perhaps, i.e. explore the space
     * @return
     */
    protected abstract Benchmark nextExplore();

    /**
     * return a previously seen classifier with a better training process (e.g. more neighbours in a KNN's
     * neighbourhood) thus exploiting the seen space further
     * @return
     */
    protected abstract Benchmark nextExploit();

    protected abstract boolean hasNextExplore();

    protected abstract boolean hasNextExploit();

    /**
     * only called when both next explore and next exploit are available
     * @return
     */
    protected abstract boolean shouldExplore();

    /**
     * Build the agent from the train data. This is helpful if the agent needs knowledge of the train data for
     * whatever reason, e.g. deducing parameter spaces.
     * @param trainData
     */
    public void buildAgent(Instances trainData) {

    }

    private boolean hasNext = false;
    private boolean hasNextCalled = false;
    private boolean explore;
    protected Set<Benchmark> fullyExploitedBenchmarks = new HashSet<>();
    protected Set<Benchmark> partiallyExploitedBenchmarks = new HashSet<>();

    @Override public List<Benchmark> getAllBenchmarks() {
        final List<Benchmark> list = new ArrayList<>();
        list.addAll(fullyExploitedBenchmarks);
        list.addAll(partiallyExploitedBenchmarks);
        return list;
    }

    @Override public final boolean hasNext() {
        if(!hasNextCalled) {
            hasNextCalled = false;
            boolean hasNextExploit = hasNextExploit();
            boolean hasNextExplore = hasNextExplore();
            if(hasNextExplore && hasNextExploit) {
                hasNextExplore = shouldExplore();
                hasNextExploit = !hasNextExplore;
            }
            if(hasNextExplore && !hasNextExploit) {
                explore = true;
            } else if(!hasNextExplore && hasNextExploit) {
                explore = false;
            } else {
                throw new IllegalStateException();
            }
            hasNext = hasNextExploit || hasNextExplore;
        }
        return hasNext;
    }

    @Override public final Benchmark next() {
        hasNextCalled = false;
        Benchmark benchmark;
        if(explore) {
            benchmark = nextExplore();
        } else {
            benchmark = nextExploit();
        }
        partiallyExploitedBenchmarks.add(benchmark);
        return benchmark;
    }

    protected abstract boolean isExploitable(Benchmark benchmark);

    @Override public boolean feedback(final Benchmark benchmark) {
        if(isExploitable(benchmark)) {
            partiallyExploitedBenchmarks.add(benchmark);
            return false;
        } else {
            partiallyExploitedBenchmarks.remove(benchmark);
            fullyExploitedBenchmarks.add(benchmark);
            return true;
        }
    }
}
