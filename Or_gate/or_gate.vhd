-- Simple OR gate design
LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
ENTITY or_gate IS
    PORT (
        a : IN STD_LOGIC;
        b : IN STD_LOGIC;
        q : OUT STD_LOGIC);
END or_gate;
ARCHITECTURE rtl OF or_gate IS
BEGIN
    PROCESS (a, b) IS
    BEGIN
        q <= a OR b;
    END PROCESS;
END rtl;