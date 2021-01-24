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
import weka.core.Randomizable;

import java.util.Random;

import static utilities.Utilities.argMax;

/**
 * Probability threshold decision maker.
 * Only makes a prediction if the highest probability is past a set threshold or the full series has been seen.
 *
 * @author Matthew Middlehurst
 */
public class ProbabilityThreshold extends EarlyDecisionMaker implements Randomizable {

    private double threshold = 0.85;

    private int finalIndex;

    private int seed = 0;
    private Random rand;

    public ProbabilityThreshold() { }

    public void setThreshold(double d) { threshold = d; }

    public void setSeed(int i) { seed = i; }

    @Override
    public int getSeed() { return seed; }

    @Override
    public void fit(Instances data, Classifier[] classifiers, int[] thresholds) {
        finalIndex = thresholds.length-1;
        rand = new Random(seed);
    }

    @Override
    public boolean decide(int thresholdIndex, double[] probabilities) {
        return (thresholdIndex == finalIndex || probabilities[argMax(probabilities, rand)] > threshold);
    }
}
