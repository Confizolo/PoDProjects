LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
ENTITY testbench IS
END testbench;
ARCHITECTURE tb OF testbench IS
    COMPONENT top IS
        PORT (
            clk : IN STD_LOGIC;
            ser_in : IN STD_LOGIC;
            ser_out : OUT STD_LOGIC
        );
    END COMPONENT;

    SIGNAL clock : STD_LOGIC := '0';
    SIGNAL ser_i : STD_LOGIC := '1';
    SIGNAL ser_o : STD_LOGIC := '0';
BEGIN
    DUT : top
    PORT MAP(
        clk => clock,
        ser_in => ser_i,
        ser_out => ser_o);

    -- clock generation
    clock <= NOT clock AFTER 10 ns;

    PROCESS
    BEGIN
        WAIT FOR 100 ns;
        -- Value 32
        WAIT FOR 17360 ns;
        ser_i <= '0';

        WAIT FOR 17360 ns;
        ser_i <= '0';
        WAIT FOR 17360 ns;
        ser_i <= '0';
        WAIT FOR 17360 ns;
        ser_i <= '0';
        WAIT FOR 17360 ns;
        ser_i <= '0';
        WAIT FOR 17360 ns;
        ser_i <= '0';
        WAIT FOR 17360 ns;
        ser_i <= '1';
        WAIT FOR 17360 ns;
        ser_i <= '0';
        WAIT FOR 17360 ns;
        ser_i <= '0';

        WAIT FOR 17360 ns;
        ser_i <= '0';
        WAIT FOR 17360 ns;
        ser_i <= '1';

        -- Value 125
        WAIT FOR 17360 ns;
        ser_i <= '0';

        WAIT FOR 17360 ns;
        ser_i <= '1';
        WAIT FOR 17360 ns;
        ser_i <= '0';
        WAIT FOR 17360 ns;
        ser_i <= '1';
        WAIT FOR 17360 ns;
        ser_i <= '1';
        WAIT FOR 17360 ns;
        ser_i <= '1';
        WAIT FOR 17360 ns;
        ser_i <= '1';
        WAIT FOR 17360 ns;
        ser_i <= '1';
        WAIT FOR 17360 ns;
        ser_i <= '0';

        WAIT FOR 17360 ns;
        ser_i <= '0';
        WAIT FOR 17360 ns;
        ser_i <= '1';
        -- Value 95
        WAIT FOR 17360 ns;
        ser_i <= '0';

        WAIT FOR 17360 ns;
        ser_i <= '1';
        WAIT FOR 17360 ns;
        ser_i <= '1';
        WAIT FOR 17360 ns;
        ser_i <= '1';
        WAIT FOR 17360 ns;
        ser_i <= '1';
        WAIT FOR 17360 ns;
        ser_i <= '1';
        WAIT FOR 17360 ns;
        ser_i <= '0';
        WAIT FOR 17360 ns;
        ser_i <= '1';
        WAIT FOR 17360 ns;
        ser_i <= '0';

        WAIT FOR 17360 ns;
        ser_i <= '0';
        WAIT FOR 17360 ns;
        ser_i <= '1';
        -- Value 56
        WAIT FOR 17360 ns;
        ser_i <= '0';

        WAIT FOR 17360 ns;
        ser_i <= '0';
        WAIT FOR 17360 ns;
        ser_i <= '0';
        WAIT FOR 17360 ns;
        ser_i <= '0';
        WAIT FOR 17360 ns;
        ser_i <= '1';
        WAIT FOR 17360 ns;
        ser_i <= '1';
        WAIT FOR 17360 ns;
        ser_i <= '1';
        WAIT FOR 17360 ns;
        ser_i <= '0';
        WAIT FOR 17360 ns;
        ser_i <= '0';

        WAIT FOR 17360 ns;
        ser_i <= '0';
        WAIT FOR 17360 ns;
        ser_i <= '1';

        WAIT;
    END PROCESS;
END tb;