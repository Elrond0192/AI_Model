<?php
/**
 * Plugin Name: Basketball Performance AI Chat
 * Plugin URI:  https://github.com/Elrond0192/AI_Model
 * Description: Embeds the Basketball Performance AI chat widget via shortcode.
 * Version:     1.0.0
 * Author:      AI_Model
 * License:     MIT
 *
 * Shortcode usage
 * ---------------
 *   [basketball_chat
 *     api_url="https://your-api.example.com"
 *     api_key=""
 *     title="Basketball AI"
 *     height="520"
 *   ]
 *
 * Parameters
 * ----------
 * api_url  (required) Full base URL of the FastAPI server,
 *          e.g. https://basketball-ai.azurewebsites.net
 *          Do NOT include a trailing slash.
 * api_key  (optional) Value for the X-API-Key header.
 *          Leave empty if API-key authentication is disabled.
 * title    (optional) Header text shown inside the widget.
 *          Default: "Basketball AI"
 * height   (optional) Widget height in pixels.
 *          Default: 520
 *
 * Installation
 * ------------
 * 1. Copy this folder (basketball-chat/) to wp-content/plugins/
 * 2. Activate the plugin in WP Admin → Plugins
 * 3. Add the shortcode to any page or post
 * 4. Make sure the FastAPI server is running and publicly reachable
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit; // Prevent direct access
}

/**
 * Register scripts and styles.
 */
function bball_chat_enqueue_assets() {
    wp_register_style(
        'basketball-chat',
        plugin_dir_url( __FILE__ ) . 'basketball-chat.css',
        [],
        '1.0.0'
    );
    wp_register_script(
        'basketball-chat',
        plugin_dir_url( __FILE__ ) . 'basketball-chat.js',
        [],
        '1.0.0',
        true   // load in footer
    );
}
add_action( 'wp_enqueue_scripts', 'bball_chat_enqueue_assets' );


/**
 * Render the chat widget shortcode.
 *
 * @param array $atts Shortcode attributes.
 * @return string     HTML output.
 */
function bball_chat_shortcode( $atts ) {
    $atts = shortcode_atts(
        [
            'api_url' => '',
            'api_key' => '',
            'title'   => 'Basketball AI',
            'height'  => '520',
        ],
        $atts,
        'basketball_chat'
    );

    // Enqueue assets only when shortcode is actually used
    wp_enqueue_style( 'basketball-chat' );
    wp_enqueue_script( 'basketball-chat' );

    $api_url  = esc_url( $atts['api_url'] );
    $api_key  = esc_attr( $atts['api_key'] );
    $title    = esc_html( $atts['title'] );
    $height   = max( 300, intval( $atts['height'] ) );

    // Unique ID so multiple widgets can coexist on the same page
    static $instance = 0;
    $instance++;
    $widget_id = 'bball-chat-' . $instance . '-' . wp_rand( 1000, 9999 );

    ob_start();
    ?>
    <div
        id="<?php echo esc_attr( $widget_id ); ?>"
        data-basketball-chat="true"
        data-api-url="<?php echo $api_url; ?>"
        data-api-key="<?php echo $api_key; ?>"
        data-title="<?php echo $title; ?>"
        style="height:<?php echo $height; ?>px;"
    ></div>
    <script>
    (function () {
        function tryInit() {
            if (typeof BasketballChat !== 'undefined') {
                BasketballChat.init(<?php echo wp_json_encode( $widget_id ); ?>);
            } else {
                setTimeout(tryInit, 50);
            }
        }
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', tryInit);
        } else {
            tryInit();
        }
    })();
    </script>
    <?php
    return ob_get_clean();
}
add_shortcode( 'basketball_chat', 'bball_chat_shortcode' );
