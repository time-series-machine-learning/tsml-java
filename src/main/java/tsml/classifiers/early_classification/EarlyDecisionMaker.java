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
package tsml.classifiers.early_classification;

import weka.classifiers.Classifier;
import weka.core.Instances;

import java.util.Arrays;
import java.util.TreeSet;

/**
 * Class for early classification decision makers.
 * When presented with a series to predict, decide on whether to make a prediction or delay until more of the series
 * is available.
 *
 * @author Matthew Middlehurst
 */
public abstract class EarlyDecisionMaker {

    protected boolean normalise = false;

    public void setNormalise(boolean b) { normalise = b; }

    public abstract void fit(Instances data, Classifier[] classifiers, int[] thresholds) throws Exception;

    public void fit(Instances data, Classifier classifier, int[] thresholds) throws Exception {
        Classifier[] classifiers = new Classifier[thresholds.length];
        Arrays.fill(classifiers, classifier);
        fit(data, classifiers, thresholds);
    }

    public void fit(Instances data, Classifier classifier) throws Exception {
        fit(data, classifier, defaultTimeStamps(data.numAttributes()-1));
    }

    public abstract boolean decide(int thresholdIndex, double[] probabilities) throws Exception;

    public int[] defaultTimeStamps(int length) {
        TreeSet<Integer> ts = new TreeSet<>();
        for (double i = 0.05; i < 0.99; i += 0.05) {
            i = Math.round(i * 100.0) / 100.0;
            int v = (int) Math.round(i * length);
            if (v >= 3)
                ts.add(v);
        }
        ts.add(length);

        int[] arr = new int[ts.size()];
        int i = 0;
        for (Integer v: ts)
            arr[i++] = v;
        return arr;
    }
}
