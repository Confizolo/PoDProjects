LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
ENTITY heartbeat IS
    PORT (clk : OUT STD_LOGIC);
END heartbeat;
ARCHITECTURE behaviour OF heartbeat IS
    CONSTANT clk_period : TIME := 10 ns;
BEGIN
    -- Clock process definition
    clk_process : PROCESS
    BEGIN
        clk <= '0';
        WAIT FOR clk_period/2;
        clk <= '1';
        WAIT FOR clk_period/2;
    END PROCESS;
END behaviour;

LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
ENTITY heartbeat_top IS
END ENTITY heartbeat_top;
ARCHITECTURE str OF heartbeat_top IS
    COMPONENT heartbeat IS
        PORT (
            clk : OUT STD_LOGIC);
    END COMPONENT heartbeat;
    SIGNAL clk : STD_LOGIC;
BEGIN -- architecture str
    DUT : ENTITY work.heartbeat
        PORT MAP(
            clk => clk);
END ARCHITECTURE str;