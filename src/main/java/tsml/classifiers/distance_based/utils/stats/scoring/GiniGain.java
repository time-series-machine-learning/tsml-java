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
 
package tsml.classifiers.distance_based.utils.stats.scoring;

import java.util.List;

public class GiniGain implements SplitScorer {
    public <A> double score(Labels<A> parent, List<Labels<A>> children) {
        return gain(parent, children, new GiniEntropy());
    }
    
    protected static <A> double weightedInverseEntropy(Labels<A> parent, List<Labels<A>> children, PartitionEntropy entropy) {
        final double parentSum = parent.getWeightSum();
        double childEntropySum = 0;
        for(Labels<A> child : children) {
            child.setLabelSet(parent.getLabelSet());
            double childEntropy = entropy.inverseEntropy(child);
            final double childSum = child.getWeightSum();
            final double proportion = childSum / parentSum;
            childEntropy *= proportion;
            childEntropySum += childEntropy;
        }
        return childEntropySum;
    }
    
    protected static <A> double weightedEntropy(Labels<A> parent, List<Labels<A>> children, PartitionEntropy entropy) {
        final double parentSum = parent.getWeightSum();
        double childEntropySum = 0;
        for(Labels<A> child : children) {
            child.setLabelSet(parent.getLabelSet());
            double childEntropy = entropy.entropy(child);
            final double childSum = child.getWeightSum();
            final double proportion = childSum / parentSum;
            childEntropy *= proportion;
            childEntropySum += childEntropy;
        }
        return childEntropySum;
    }
    
    protected static <A> double gain(Labels<A> parent, List<Labels<A>> children, PartitionEntropy entropy) {
        final double parentEntropy = entropy.entropy(parent);
        double childEntropySum = weightedEntropy(parent, children, entropy);
        return parentEntropy - childEntropySum;
    }
    
}
