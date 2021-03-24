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
 
package tsml.classifiers.distance_based.utils.collections.tree;

import java.io.Serializable;
import java.util.List;

/**
 *
 *  Purpose: a tree data structure. Imposes no ideology regarding context of nodes, e.g. sorted / balanced, etc. 
 *  This solely provide structural implementation and leave the responsibility of maintaining a coherent context to 
 *  an implementing class.
 * <p>
 * Contributors: goastler
 */

public interface Tree<A> extends Serializable, List<A> {

    TreeNode<A> getRoot();

    void setRoot(TreeNode<A> root);

    /**
     * Return the number of nodes in the tree
     * @return
     */
    default int size() {
        TreeNode<A> root = getRoot();
        if(root == null) {
            return 0;
        }
        return root.size();
    }
    
    /**
     * Return the max depth. This is measured with root node == height 0.
     * @return
     */
    default int height() {
        TreeNode<A> root = getRoot();
        if(root == null) {
            return 0;
        }
        return root.height();
    }

    default void clear() {
        setRoot(null);
    }

    default int nodeCount() {
        TreeNode<A> root = getRoot();
        if(root == null) {
            return 0;
        }
        return root.numNodes();
    }
    
}
