LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
ENTITY top IS
    PORT (
        clk : IN STD_LOGIC;
        ser_in : IN STD_LOGIC;
        ser_out : OUT STD_LOGIC
    );
END top;
ARCHITECTURE rtl OF top IS
    COMPONENT fir_filter IS
        PORT (
            i_clk : IN STD_LOGIC;
            i_rstb : IN STD_LOGIC;
            -- data input
            i_valid : IN STD_LOGIC;
            i_data : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
            -- filtered data 
            o_valid : OUT STD_LOGIC;
            o_data : OUT STD_LOGIC_VECTOR(7 DOWNTO 0));
    END COMPONENT;

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
    SIGNAL rx_valid : STD_LOGIC := '0';
    SIGNAL tx_valid : STD_LOGIC := '0';
    SIGNAL tx_data : STD_LOGIC_VECTOR(7 DOWNTO 0);
    SIGNAL rx_data : STD_LOGIC_VECTOR(7 DOWNTO 0);
    SIGNAL busy : STD_LOGIC := '0';
    SIGNAL rst : STD_LOGIC := '1';
BEGIN
    transmitter : uart_t
    PORT MAP(
        txclk => clk,
        tx_data => tx_data,
        valid => tx_valid,
        tx_busy => busy,
        uart_out => ser_out
    );
    receiver : uart_r
    PORT MAP(
        clk => clk,
        uart_rx => ser_in,
        valid => rx_valid,
        data => rx_data
    );
    fir : fir_filter
    PORT MAP(
        i_clk => clk,
        i_rstb => rst,
        -- data 
        i_valid => rx_valid,
        i_data => rx_data,
        -- filte
        o_valid => tx_valid,
        o_data => tx_data
    );
END ARCHITECTURE rtl;