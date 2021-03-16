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

import weka.classifiers.AbstractClassifier;

/**
 * Abstract classifier class for early classification classifiers.
 *
 * @author Matthew Middlehurst
 */
public abstract class AbstractEarlyClassifier extends AbstractClassifier {

    /** Time point thresholds for classifiers to make predictions at */
    protected int[] thresholds;

    protected boolean normalise = false;

    public int[] getThresholds(){ return thresholds; }

    public void setThresholds(int[] t){ thresholds = t; }

    public void setNormalise(boolean b) { normalise = b; }
}
