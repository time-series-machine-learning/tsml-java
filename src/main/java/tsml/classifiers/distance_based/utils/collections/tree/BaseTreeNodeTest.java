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

import org.junit.Assert;
import org.junit.Test;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class BaseTreeNodeTest {

    @Test
    public void testHeightAndSize() {
        //  a
        //  |
        //  b
        // / \
        // c d
        // |
        // e
        TreeNode<String> a = new BaseTreeNode<>("a");
        TreeNode<String> b = new BaseTreeNode<>("b");
        TreeNode<String> c = new BaseTreeNode<>("c");
        TreeNode<String> d = new BaseTreeNode<>("d");
        TreeNode<String> e = new BaseTreeNode<>("e");
        a.addChild(b);
        b.addChild(c);
        b.addChild(d);
        c.addChild(e);
        //            e.addChild(e); // todo check this errs
        Assert.assertEquals(a.size(), 5);
        Assert.assertEquals(a.height(), 4);
    }

    //        @Test(expected = IllegalArgumentException.class)
    //        public void testAddSelfAsChild() {
    //            BaseTreeNode<String> node = new BaseTreeNode<>("a");
    //            node.addChild(node);
    //        }
}
