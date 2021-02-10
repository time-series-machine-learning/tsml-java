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
import java.util.Collection;
import java.util.List;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */

public interface TreeNode<A> extends Serializable, Collection<TreeNode<A>> {

    TreeNode<A> getParent();

    void setParent(TreeNode<A> parent);

    List<TreeNode<A>> getChildren();

    default boolean hasChildren() {
        return getChildren().size() > 0;
    }

    A getElement();

    void setElement(A element);

    boolean hasElement();

    int numChildren();

    int size();

    boolean isLeaf();

    int height();

    int getLevel();

    boolean addChild(TreeNode<A> child);

    boolean removeChild(TreeNode<A> child);

    boolean isRoot();
}
