package utilities.samplers;

import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static utilities.InstanceTools.indexByClass;
import static utilities.InstanceTools.instancesByClass;

public class RandomRoundRobinIndexSampler implements Sampler{

    private List<List<Integer>> instancesByClass;
    private Random random;
    private final List<Integer> indicies = new ArrayList<>();

    public RandomRoundRobinIndexSampler(Random random){
        this.random = random;
    }

    public RandomRoundRobinIndexSampler(){
        random = new Random();
    }

    private void regenerateClassValues() {
        for(int i = 0; i < instancesByClass.size(); i++) {
            indicies.add(i);
        }
    }

    public void setInstances(Instances instances) {
        instancesByClass = indexByClass(instances);
        regenerateClassValues();
    }

    public boolean hasNext() {
        return !indicies.isEmpty() || !instancesByClass.isEmpty();
    }

    public Integer next() {
        int classValue = indicies.remove(random.nextInt(indicies.size()));
        List<Integer> homogeneousInstances = instancesByClass.get(classValue);
        int instance = homogeneousInstances.remove(random.nextInt(homogeneousInstances.size()));
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
