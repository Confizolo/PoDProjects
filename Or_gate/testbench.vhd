LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
ENTITY testbench IS
END testbench;
ARCHITECTURE tb OF testbench IS
    COMPONENT or_gate IS
        PORT (
            a : IN STD_LOGIC;
            b : IN STD_LOGIC;
            q : OUT STD_LOGIC);
    END COMPONENT;
    SIGNAL a_in, b_in, q_out : STD_LOGIC;
BEGIN
    DUT : or_gate PORT MAP(a_in, b_in, q_out);
    PROCESS
    BEGIN
        a_in <= '0';
        b_in <= '0';
        WAIT FOR 1 ns;
        ASSERT(q_out = '0') REPORT
        "Fail 0/0" SEVERITY error;
        a_in <= '0';
        b_in <= '1';
        WAIT FOR 1 ns;
        ASSERT(q_out = '1') REPORT
        "Fail 0/1" SEVERITY error;
        a_in <= '1';
        b_in <= 'X';
        WAIT FOR 1 ns;
        ASSERT(q_out = '1') REPORT
        "Fail 1/X" SEVERITY error;
        a_in <= '1';
        b_in <= '1';
        WAIT FOR 1 ns;
        ASSERT(q_out = '1') REPORT
        "Fail 1/1" SEVERITY error;
        -- Clear inputs
        a_in <= '0';
        b_in <= '0';
        ASSERT false REPORT "Test done." SEVERITY
        note;
        WAIT;
    END PROCESS;
END tb;