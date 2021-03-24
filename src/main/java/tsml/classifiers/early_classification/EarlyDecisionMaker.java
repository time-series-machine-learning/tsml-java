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
        int[] ts = new int[20];
        ts[19] = length;
        for (int i = 0; i < 19; i++){
            ts[i] = (int)Math.round((i+1) * 0.05 * length);
        }
        return ts;
    }
}
