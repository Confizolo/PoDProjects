LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
ENTITY patterndetect_tb IS
END ENTITY patterndetect_tb;
ARCHITECTURE test OF patterndetect_tb IS
    SIGNAL a
    : STD_LOGIC;
    SIGNAL clk : STD_LOGIC := '0';
    SIGNAL rst : STD_LOGIC;
    SIGNAL y
    : STD_LOGIC;
BEGIN -- architecture test
    DUT : ENTITY work.patterndetect
        PORT MAP(
            a
            => a,
            clk => clk,
            rst => rst,
            y
            => y);
    clk <= NOT clk AFTER 2 ns;
    WaveGen_Proc : PROCESS
    BEGIN
        a
        <= '0';
        rst <= '1';
        WAIT FOR 10 ns;
        WAIT UNTIL rising_edge(clk);
        rst <= '0';
        WAIT FOR 10 ns;
        WAIT UNTIL rising_edge(clk);
        rst <= '1';
        WAIT UNTIL rising_edge(clk);
        a <= '0';
        WAIT UNTIL rising_edge(clk);
        a <= '1';
        WAIT UNTIL rising_edge(clk);
        a <= '0';
        WAIT UNTIL rising_edge(clk);
        a <= '1';
        WAIT FOR 100 ns;
        WAIT;
    END PROCESS WaveGen_Proc;
END ARCHITECTURE test;