package evaluation.tuning.searchers;

import evaluation.tuning.ParameterSet;
import weka.classifiers.functions.GaussianProcesses;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.function.Function;

//Needs to be improved on, not currently used in any classifier. Waiting for Aaron's version to make improvements.
public class BayesianSearcher extends ParameterSearcher{

    private GaussianProcesses gp = new GaussianProcesses();
    private Function<ParameterSet, Double> objectiveFunction;
    private int maxIterations = 100;
    private int numSeedPoints = 20;

    private String[] keys;
    private List<String>[] values;
    private Instance bestParameters;

    public BayesianSearcher(Function<ParameterSet, Double> objectiveFunction){
        this.objectiveFunction = objectiveFunction;
        gp.setKernel(new RBFKernel());
        try {
            gp.setOptions(new String[]{"-N","2"});
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public ParameterSet getBestParameters(){ return instanceToPSet(bestParameters); }

    public void setKernel(Kernel kernel){ gp.setKernel(kernel); }

    public void setNoise(double noise) { gp.setNoise(noise); }

    public void setMaxIterations(int max) { maxIterations = max; }

    @Override
    public Iterator<ParameterSet> iterator() { return new BayesianSearchIterator(); }

    private ParameterSet instanceToPSet(Instance inst){
        ParameterSet pset = new ParameterSet();
        for (int i = 0; i < inst.numAttributes()-1; i++) {
            pset.parameterSet.put(keys[i], values[i].get((int)inst.value(i)));
        }
        return pset;
    }

    public class BayesianSearchIterator implements Iterator<ParameterSet> {

        private boolean improvementExpected = true;
        private Instances parameterPool;
        private Instances pastParameters;
        private Instance chosenParameters;
        private double maxObjVal = 0;
        private int numIterations = 0;

        private Random rand;

        public BayesianSearchIterator(){
            rand = new Random(seed);

            keys = new String[space.numParas()];
            values = new List[space.numParas()];
            int g = 0;

            for (Map.Entry<String, List<String>> entry : space.parameterLists.entrySet()) {
                keys[g] = entry.getKey();
                values[g] = entry.getValue();
                g++;
            }

            int numAtts = keys.length+1;
            ArrayList<Attribute> atts = new ArrayList<>(numAtts);
            for (int i = 0; i < numAtts; i++){
                atts.add(new Attribute("att" + i));
            }

            parameterPool = new Instances("Parameters", atts, 0);
            parameterPool.setClassIndex(parameterPool.numAttributes()-1);
            pastParameters = new Instances(parameterPool, 0);
            pastParameters.setClassIndex(pastParameters.numAttributes()-1);

            GridSearcher gs = new GridSearcher();
            gs.space = space;

            for (ParameterSet p : gs) {
                double[] idx = new double[keys.length + 1];

                int i = 0;
                for (Map.Entry<String, String> entry : p.parameterSet.entrySet()) {
                    idx[i] = values[i].indexOf(entry.getValue());
                    i++;
                }

                DenseInstance inst = new DenseInstance(1, idx);
                parameterPool.add(inst);
            }
        }

        @Override
        public boolean hasNext() {
            return improvementExpected;
        }

        @Override
        public ParameterSet next() {
            ParameterSet pset = null;

            if (numIterations < numSeedPoints){
                chosenParameters = parameterPool.remove(rand.nextInt(parameterPool.size()));

                pset = instanceToPSet(chosenParameters);
                double objVal = objectiveFunction.apply(pset);
                chosenParameters.setValue(chosenParameters.classIndex(), objVal);
                pastParameters.add(chosenParameters);

                if (objVal > maxObjVal){
                    maxObjVal = objVal;
                    bestParameters = chosenParameters;
                }

            }
            else{
                try {
                    gp.buildClassifier(pastParameters);

                    double maxVal = 0;

                    for (Instance inst : parameterPool) {
                        double pred = gp.classifyInstance(inst);

                        if (pred - maxObjVal > maxVal) {
                            maxVal = pred - maxObjVal;
                            chosenParameters = inst;
                        }
                    }

                    //Expected improvement, probably broken

//                    for (Instance inst: parameterPool){
//                        double mean = gp.classifyInstance(inst);
//                        double std = gp.getStandardDeviation(inst); //different from sktime std
//
//                        if (std != 0){
//                            NormalDistribution n = new NormalDistribution();
//                            double imp = (mean - maxObjVal - 0.01);
//                            double z = imp / std;
//                            double ei = imp * n.cumulativeProbability(z) + std * n.density(z);
//
//                            if (ei > maxVal){
//                                maxVal = ei;
//                                chosenParameters = inst;
//                            }
//                        }
//                    }

                    if (maxVal == 0 || numIterations == maxIterations){
                        improvementExpected = false;
                        chosenParameters = bestParameters;
                        pset = instanceToPSet(chosenParameters);
                    }
                    else{
                        pset = instanceToPSet(chosenParameters);
                        double objVal = objectiveFunction.apply(pset);
                        chosenParameters.setValue(chosenParameters.classIndex(), objVal);
                        pastParameters.add(chosenParameters);
                        parameterPool.remove(chosenParameters);

                        if (objVal > maxObjVal){
                            maxObjVal = objVal;
                            bestParameters = chosenParameters;
                        }
                    }
                }
                catch (Exception e){
                    e.printStackTrace();
                }
            }

            numIterations++;

            return pset;
        }
    }
}
