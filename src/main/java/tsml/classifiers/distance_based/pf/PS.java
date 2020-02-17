package tsml.classifiers.distance_based.pf;
/*

purpose: // todo - docs - type the purpose of the code here

created edited by goastler on 17/02/2020
    
*/

import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.MemoryWatchable;
import tsml.classifiers.TrainTimeContractable;
import utilities.MemoryWatcher;
import utilities.Rand;
import utilities.StopWatch;
import utilities.iteration.RandomListIterator;
import utilities.serialisation.SerFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

public class PS extends EnhancedAbstractClassifier implements Rand, TrainTimeContractable {

    public PS() {

    }

    public PS(PS other) {
        throw new UnsupportedOperationException();
    }

    private List<Instances> partitions; // partitions of data
    private List<Instance> exemplars;
    private Instances data;
    private Iterator<Instance> iterator;
    private StopWatch trainTimer = new StopWatch();
    private MemoryWatcher memoryWatcher = new MemoryWatcher();
    private SerFunction<Instances, Iterator<Instance>> iteratorBuilder = (SerFunction<Instances, Iterator<Instance>>) instances -> new RandomListIterator<>(instances, rand.nextInt());

    @Override
    public void buildClassifier(Instances trainData) throws Exception {
        if(rebuild) {
            memoryWatcher.resetAndEnable();
            trainTimer.resetAndEnable();
        }
        super.buildClassifier(trainData);
        this.data = trainData;
        if(rebuild) {
            rebuild = false;
            iterator = iteratorBuilder.apply(data);
        }
    }

    public boolean hasNext() {
        return iterator.hasNext() && hasRemainingTraining();
    }

    public PS next() { // todo should this be public
        Instance instance = iterator.next();

    }

    public void cleanUp() {
        data = null;
        partitions = null;
    }

    @Override
    public String toString() {
        return getClass().getSimpleName();
    }

    @Override
    public void setRandom(Random random) {
        this.rand = random;
    }

    @Override
    public Random getRandom() {
        return rand;
    }
}
