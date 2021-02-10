/*
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
 
package tsml.classifiers.distance_based.knn;

import evaluation.storage.ClassifierResults;
import tsml.classifiers.EnhancedAbstractClassifier;
import weka.classifiers.Classifier;

import java.util.Objects;

public class Benchmark {
    private double score = -1;
    private EnhancedAbstractClassifier classifier;
    private static int idCounter = 0;
    private final int id = idCounter++;

    public Benchmark(EnhancedAbstractClassifier classifier) {
        setClassifier(classifier);
    }

    public EnhancedAbstractClassifier getClassifier() {
        return classifier;
    }

    public void setClassifier(final EnhancedAbstractClassifier classifier) {
        this.classifier = classifier;
    }

    public double getScore() {
        return score;
    }

    public void setScore(final double score) {
        this.score = score;
    }

    @Override public boolean equals(final Object o) {
        if(this == o) {
            return true;
        }
        if(!(o instanceof Benchmark)) {
            return false;
        }
        final Benchmark benchmark = (Benchmark) o;
        return id == benchmark.id;
    }

    @Override public int hashCode() {
        return id;
    }
}
