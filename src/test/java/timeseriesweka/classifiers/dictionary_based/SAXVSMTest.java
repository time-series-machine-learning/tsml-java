/*
 * Copyright (C) 2019 xmw13bzu
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package timeseriesweka.classifiers.dictionary_based;

import java.lang.invoke.MethodHandles;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;
import utilities.ClassifierTools;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class SAXVSMTest {
     static String cleanClassNameString = MethodHandles.lookup().lookupClass().getSimpleName().replace("Test", "");
    
    public SAXVSMTest() {
    }
    
    @BeforeClass
    public static void setUpClass() {
        System.out.println("-----Start " + cleanClassNameString + " tests-----");
    }
    
    @AfterClass
    public static void tearDownClass() {
        System.out.println("-----End " + cleanClassNameString + " tests-----\n");
    }
    
    @Before
    public void setUp() {
    }
    
    @After
    public void tearDown() {
    }

    /**
     * Test of simple results reproduction.
     */
    @Test
    public void testReproduction() throws Exception {
        System.out.println("--testReproduction()");
        assertTrue(ClassifierTools.testUtils_confirmIPDReproduction(new SAXVSM(), 0.7580174927113703, "2019/09/26"));
    }
}
