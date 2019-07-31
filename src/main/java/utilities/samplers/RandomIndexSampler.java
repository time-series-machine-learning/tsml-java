package utilities.samplers;

import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class RandomIndexSampler implements Sampler{

    private List<Integer> instances;
    private Random random;

    public RandomIndexSampler(Random random){
        this.random = random;
    }

    public RandomIndexSampler(){
        random = new Random();
    }

    public void setInstances(Instances instances) {
        this.instances = new ArrayList(instances.numInstances());
        for (int i = 0; i < instances.numInstances(); i++){
            this.instances.add(i);
        }
    }

    public boolean hasNext() { return !instances.isEmpty(); }

    public Integer next() { return instances.remove(random.nextInt(instances.size())); }
}
