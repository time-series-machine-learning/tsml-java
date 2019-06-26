package utilities.samplers;

import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static utilities.InstanceTools.instancesByClass;

public class RandomRoundRobinSampler implements Sampler{

    private List<Instances> instancesByClass;
    private Random random;
    private final List<Integer> indicies = new ArrayList<>();

    public RandomRoundRobinSampler(Random random){
        this.random = random;
    }

    public RandomRoundRobinSampler(){
        random = new Random();
    }

    private void regenerateClassValues() {
        for(int i = 0; i < instancesByClass.size(); i++) {
            indicies.add(i);
        }
    }

    public void setInstances(Instances instances) {
        instancesByClass = instancesByClass(instances);
        regenerateClassValues();
    }

    public boolean hasNext() {
        return !indicies.isEmpty() || !instancesByClass.isEmpty();
    }

    public Instance next() {
        int classValue = indicies.remove(random.nextInt(indicies.size()));
        Instances homogeneousInstances = instancesByClass.get(classValue);
        Instance instance = homogeneousInstances.remove(random.nextInt(homogeneousInstances.size()));
        if(homogeneousInstances.isEmpty()) {
            instancesByClass.remove(classValue);
            for(int i = 0; i < indicies.size(); i++) {
                if (indicies.get(i) > classValue) {
                    indicies.set(i, indicies.get(i) - 1);
                }
            }
        }
        if(indicies.isEmpty()) {
            regenerateClassValues();
        }
        return instance;
    }
}
