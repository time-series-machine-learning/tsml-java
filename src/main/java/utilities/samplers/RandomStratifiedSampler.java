package utilities.samplers;

import utilities.ArrayUtilities;
import weka.core.Instance;
import weka.core.Instances;
import java.util.List;
import java.util.Random;
import static utilities.InstanceTools.classDistribution;
import static utilities.InstanceTools.instancesByClass;
import static utilities.Utilities.argMax;

public class RandomStratifiedSampler implements Sampler{

    private List<Instances> instancesByClass;
    private double[] classDistribution;
    private double[] classSamplingProbabilities;
    private int count;
    private Random random;
    private int maxCount;

    public RandomStratifiedSampler(Random random){
        this.random = random;
    }

    public RandomStratifiedSampler(){
        random = new Random();
    }

    public void setInstances(Instances instances) {
        instancesByClass = instancesByClass(instances);
        classDistribution = classDistribution(instances);
        classSamplingProbabilities = classDistribution(instances);
        count = 0;
        maxCount = instances.size();
    }

    public boolean hasNext() {
        return count < maxCount;
    }

    public Instance next() {
        int sampleClass = argMax(classSamplingProbabilities, random);
        Instances homogeneousInstances = instancesByClass.get(sampleClass); // instances of the class value
        Instance sampledInstance = homogeneousInstances.remove(random.nextInt(homogeneousInstances.numInstances()));
        classSamplingProbabilities[sampleClass]--;
        ArrayUtilities.add(classSamplingProbabilities, classDistribution);
        count++;
        return sampledInstance;
    }
}
