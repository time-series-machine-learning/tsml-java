package utilities.samplers;

import utilities.ArrayUtilities;
import weka.core.Instances;

import java.util.List;
import java.util.Random;

import static utilities.InstanceTools.*;
import static utilities.Utilities.argMax;

public class RandomStratifiedIndexSampler implements Sampler{

    private List<List<Integer>> instancesByClass;
    private double[] classDistribution;
    private double[] classSamplingProbabilities;
    private int count;
    private Random random;
    private int maxCount;

    public RandomStratifiedIndexSampler(Random random){
        this.random = random;
    }

    public RandomStratifiedIndexSampler(){
        random = new Random();
    }

    public void setInstances(Instances instances) {
        instancesByClass = indexByClass(instances);
        classDistribution = classDistribution(instances);
        classSamplingProbabilities = classDistribution(instances);
        count = 0;
        maxCount = instances.size();
    }

    public boolean hasNext() {
        return count < maxCount;
    }

    public Integer next() {
        int sampleClass = argMax(classSamplingProbabilities, random);
        List<Integer> homogeneousInstances = instancesByClass.get(sampleClass); // instances of the class value
        int sampledInstance = homogeneousInstances.remove(random.nextInt(homogeneousInstances.size()));
        classSamplingProbabilities[sampleClass]--;
        ArrayUtilities.add(classSamplingProbabilities, classDistribution);
        return sampledInstance;
    }
}
