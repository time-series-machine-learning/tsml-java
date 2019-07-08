package utilities.samplers;

import weka.core.Instance;
import weka.core.Instances;

import java.util.List;
import java.util.Random;

public class RandomSampler implements Sampler{

    private Instances instances;
    private Random random;

    public RandomSampler(Random random){
        this.random = random;
    }

    public RandomSampler(){
        random = new Random();
    }

    public void setInstances(Instances instances) { this.instances = new Instances(instances); }

    public boolean hasNext() { return !instances.isEmpty(); }

    public Instance next() { return instances.remove(random.nextInt(instances.size())); }
}
