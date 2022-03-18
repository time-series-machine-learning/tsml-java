/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package evaluation.tuning.searchers;

import evaluation.tuning.ParameterSet;
import statistics.distributions.NormalDistribution;
import weka.classifiers.functions.GaussianProcesses;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.function.Function;

public class BayesianSearcher extends ParameterSearcher {

    private GaussianProcesses gp = new GaussianProcesses();
    private Function<ParameterSet, Double> objectiveFunction;
    private int maxIterations = 500;
    private int numSeedPoints = 50;

    private String[] keys;
    private List<String>[] values;
    private Instance bestParameters;

    public BayesianSearcher(Function<ParameterSet, Double> objectiveFunction) throws Exception {
        this.objectiveFunction = objectiveFunction;
        gp.setKernel(new RBFKernel());
        gp.setNoise(2);

        throw new Exception("Currently \"in progress\", most likely broken.");
    }

    public ParameterSet getBestParameters(){ return instanceToParameterSet(bestParameters); }

    public void setKernel(Kernel kernel){ gp.setKernel(kernel); }

    public void setNoise(double noise) { gp.setNoise(noise); }

    public void setMaxIterations(int max) { maxIterations = max; }

    @Override
    public Iterator<ParameterSet> iterator() { return new BayesianSearchIterator(); }

    private ParameterSet instanceToParameterSet(Instance inst){
        ParameterSet pset = new ParameterSet();
        for (int i = 0; i < inst.numAttributes()-1; i++) {
            pset.parameterSet.put(keys[i], values[i].get((int)inst.value(i)));
        }
        return pset;
    }

    private class BayesianSearchIterator implements Iterator<ParameterSet> {

        private boolean improvementExpected = true;
        private Instances parameterPool;
        private Instances pastParameters;
        private double maxObjVal = 0;
        private int numIterations = 0;

        private Random rand;

        public BayesianSearchIterator(){
            rand = new Random(seed);

            keys = new String[space.numParas()];
            values = new List[space.numParas()];
            bestParameters = null;

            int n = 0;
            for (Map.Entry<String, List<String>> entry : space.parameterLists.entrySet()) {
                keys[n] = entry.getKey();
                values[n] = entry.getValue();
                n++;
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
            Instance chosenParameters = null;

            if (numIterations < numSeedPoints){
                chosenParameters = parameterPool.remove(rand.nextInt(parameterPool.size()));

                pset = instanceToParameterSet(chosenParameters);
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

                    //Expected improvement, probably broken
                    for (Instance inst: parameterPool){
                        double mean = gp.classifyInstance(inst);
                        double std = gp.getStandardDeviation(inst);

                        if (std != 0){
                            NormalDistribution n = new NormalDistribution();
                            double imp = (mean - maxObjVal - 0.01);
                            double z = imp / std;
                            double ei = imp * n.getCDF(z) + std * n.getDensity(z);

                            if (ei > maxVal){
                                maxVal = ei;
                                chosenParameters = inst;
                            }
                        }
                    }

                    if (maxVal == 0 || numIterations == maxIterations){
                        improvementExpected = false;
                        chosenParameters = bestParameters;
                        pset = instanceToParameterSet(chosenParameters);
                    }
                    else{
                        pset = instanceToParameterSet(chosenParameters);
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
