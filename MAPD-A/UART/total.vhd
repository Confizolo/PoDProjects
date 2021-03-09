LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
ENTITY total IS
    PORT (
        clk, valid_in : IN STD_LOGIC;
        tx_data : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
        tx_busy : OUT STD_LOGIC;
        g_out : OUT STD_LOGIC_VECTOR(7 DOWNTO 0)
    );
END total;
ARCHITECTURE rtl OF total IS
    COMPONENT uart_r IS
        PORT (
            clk, uart_rx : IN STD_LOGIC;
            valid : OUT STD_LOGIC;
            data : OUT STD_LOGIC_VECTOR(7 DOWNTO 0)
        );
    END COMPONENT;
    COMPONENT uart_t IS
        PORT (
            txclk, valid : IN STD_LOGIC;
            tx_data : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
            uart_out, tx_busy : OUT STD_LOGIC
        );
    END COMPONENT;
    SIGNAL uart_rx : STD_LOGIC;
    SIGNAL valid_s : STD_LOGIC;
BEGIN
    gen : uart_t
    PORT MAP(
        txclk => clk,
        valid => valid_in,
        tx_data => tx_data,
        uart_out => uart_rx,
        tx_busy => tx_busy
    );
    receiv : uart_r
    PORT MAP(
        clk => clk,
        uart_rx => uart_rx,
        valid => valid_s,
        data => g_out
    );
END ARCHITECTURE rtl;