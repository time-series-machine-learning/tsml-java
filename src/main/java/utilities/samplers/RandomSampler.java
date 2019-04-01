package utilities.samplers;

import weka.core.Instance;
import weka.core.Instances;

import java.util.List;
import java.util.Random;

public class RandomSampler{

    private Instances instances;
    private Random random = new Random();

    public RandomSampler(Random random){
        this.random = random;
    }

    public RandomSampler(){
        random = new Random();
    }

    public void setInstances(Instances instances) { this.instances = instances; }

    public boolean hasNext() { return !instances.isEmpty(); }

    public Instance next() { return instances.remove(random.nextInt(instances.size())); }
}
