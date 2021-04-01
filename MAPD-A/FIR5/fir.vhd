LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
USE ieee.numeric_std.ALL;

ENTITY fir_filter_5 IS
    PORT (
        i_clk : IN STD_LOGIC;
        i_rstb : IN STD_LOGIC;
        -- coefficient
        i_coeff_0 : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
        i_coeff_1 : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
        i_coeff_2 : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
        i_coeff_3 : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
        i_coeff_4 : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
        -- data input
        i_data : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
        -- filtered data 
        o_data : OUT STD_LOGIC_VECTOR(7 DOWNTO 0));
END fir_filter_5;

ARCHITECTURE rtl OF fir_filter_5 IS

    TYPE t_data_pipe IS ARRAY (0 TO 4) OF signed(7 DOWNTO 0);
    TYPE t_coeff IS ARRAY (0 TO 4) OF signed(7 DOWNTO 0);

    TYPE t_mult IS ARRAY (0 TO 4) OF signed(15 DOWNTO 0);
    TYPE t_add_st0 IS ARRAY (0 TO 1) OF signed(15 + 1 DOWNTO 0);

    SIGNAL r_coeff : t_coeff;
    SIGNAL p_data : t_data_pipe;
    SIGNAL r_mult : t_mult;
    SIGNAL r_add_st0 : t_add_st0;
    SIGNAL r_add_st1 : signed(15 + 2 DOWNTO 0);
    SIGNAL r_add_st2 : signed(15 + 3 DOWNTO 0);

BEGIN

    p_input : PROCESS (i_rstb, i_clk)
    BEGIN
        IF (i_rstb = '0') THEN
            p_data <= (OTHERS => (OTHERS => '0'));
            r_coeff <= (OTHERS => (OTHERS => '0'));
            ELSIF (rising_edge(i_clk)) THEN
            p_data <= signed(i_data) & p_data(0 TO p_data'length - 2);
            r_coeff(0) <= signed(i_coeff_0);
            r_coeff(1) <= signed(i_coeff_1);
            r_coeff(2) <= signed(i_coeff_2);
            r_coeff(3) <= signed(i_coeff_3);
            r_coeff(4) <= signed(i_coeff_4);
        END IF;
    END PROCESS p_input;

    p_mult : PROCESS (i_rstb, i_clk)
    BEGIN
        IF (i_rstb = '0') THEN
            r_mult <= (OTHERS => (OTHERS => '0'));
            ELSIF (rising_edge(i_clk)) THEN
            FOR k IN 0 TO 4 LOOP
                r_mult(k) <= p_data(k) * r_coeff(k);
            END LOOP;
        END IF;
    END PROCESS p_mult;

    p_add_st0 : PROCESS (i_rstb, i_clk)
    BEGIN
        IF (i_rstb = '0') THEN
            r_add_st0 <= (OTHERS => (OTHERS => '0'));
            ELSIF (rising_edge(i_clk)) THEN
            FOR k IN 0 TO 1 LOOP
                r_add_st0(k) <= resize(r_mult(2 * k), 17) + resize(r_mult(2 * k + 1), 17);
            END LOOP;
        END IF;
    END PROCESS p_add_st0;

    p_add_st1 : PROCESS (i_rstb, i_clk)
    BEGIN
        IF (i_rstb = '0') THEN
            r_add_st1 <= (OTHERS => '0');
            ELSIF (rising_edge(i_clk)) THEN
            r_add_st1 <= resize(r_add_st0(0), 18) + resize(r_add_st0(1), 18);
        END IF;
    END PROCESS p_add_st1;

    p_add_st2 : PROCESS (i_rstb, i_clk)
    BEGIN
        IF (i_rstb = '0') THEN
            r_add_st2 <= (OTHERS => '0');
            ELSIF (rising_edge(i_clk)) THEN
            r_add_st2 <= resize(r_add_st1, 19) + resize(r_mult(4), 19);
        END IF;
    END PROCESS p_add_st2;

    p_output : PROCESS (i_rstb, i_clk)
    BEGIN
        IF (i_rstb = '0') THEN
            o_data <= (OTHERS => '0');
            ELSIF (rising_edge(i_clk)) THEN
            o_data <= STD_LOGIC_VECTOR(r_add_st2(15 DOWNTO 8));
        END IF;
    END PROCESS p_output;
END rtl;